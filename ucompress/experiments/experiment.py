from .solution import Solution
import numpy as np
from ucompress.experiments.cheb import cheb


class Experiment():
    """
    Superclass that contains common attributes and methods
    for the two experiment subclasses
    """

    def __init__(self, model, pars):
        """
        Constructor
        """
        self.mech = model["mechanics"]
        self.perm = model["permeability"]
        self.pars = pars

        self.preallocate()

        # set default solver options
        self.solver_opts = {
            "monitor_convergence": False, # monitor convergence of newton iterations
            "newton_max_iterations": 10, # maximum number of newton iterations
            "newton_tol": 1e-6 # newton convergence tolerance
        }

    def preallocate(self):
        """
        Preallocates variables that are common to both solvers
        """

        # number of grid points
        N = self.pars.computational["N"]
        self.N = N

        # indices
        self.ind_u = np.arange(N)
        self.ind_p = np.arange(N, 2*N)
        self.ind_l = 2*N

        # build operators
        D, y = cheb(self.N)
        self.D = 2 * D
        self.r = (y + 1) / 2

        self.I = np.eye(N)

        # preallocation of arrays for residuals
        self.F_u = np.zeros(N)
 
        # preallocation of common Jacobian entries
        self.J_uu = np.zeros((N, N))

        # imposing BCs in Jacobian
        self.J_uu[0,0] = 1

        # operator for pressure
        self.J_pp = self.D.copy()
        self.J_pp[-1,:] = 0; self.J_pp[-1, -1] = 1

        # derivatives of stretches wrt u
        self.lam_r_u = self.D
        self.lam_t_u = np.zeros((N, N))
        self.lam_t_u[0,:] = D[0, :]
        self.lam_t_u[1:,1:] = np.diag(1 / self.r[1:])

        # weights for the trapezoidal rule
        self.w = np.zeros(N)
        dr = np.diff(self.r)
        self.w[0] = dr[0] / 2
        self.w[-1] = dr[-1] / 2
        self.w[1:-1] = (dr[1:] + dr[:-1]) / 2



    def compute_stretches(self, u):
        """
        Computes the radial and orthoradial stretches
        """
        lam_r = 1 + self.D @ u
        lam_t = np.r_[1 + self.D[0,:] @ u, 1 + u[1:] / self.r[1:]]

        return lam_r, lam_t


    def compute_J(self):
        """
        Computes the Jacobian (J = det(F)) and its derivatives
        """
        self.J = self.lam_r * self.lam_t * self.lam_z
        self.J_u = self.lam_z * (np.diag(self.lam_t) @ self.lam_r_u + np.diag(self.lam_r) @ self.lam_t_u)
        self.J_l = self.lam_r * self.lam_t


    def compute_pressure(self):
        """
        Computes the pressure if the deformation is known
        """
        k = self.perm.eval_permeability(self.J)
        rhs = self.r * self.lam_r**2 / 2 / k / self.J / self.dt * (self.lam_t**2 * self.lam_z - self.lam_t_old**2 * self.lam_z_old)
        rhs[-1] = 0
        self.p = np.linalg.solve(self.J_pp, rhs)


    def compute_force(self):
        """
        Computes the force on the platten if the deformation
        and pressure are known using the trapezoidal rule
        """

        _, _, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)

        self.F = 2 * np.pi * np.sum(self.w * (S_z - self.p * self.lam_r * self.lam_t) * self.r)


    def newton_iterations(self, X):
        """
        Implementation of Newton's method
        """

        conv = False
        for n in range(self.solver_opts["newton_max_iterations"]):

            # extract solution components
            self.u = X[self.ind_u]

            if self.loading == 'force':
                self.p = X[self.ind_p]
                self.lam_z = X[self.ind_l]

            # compute new stretches and Jacobian
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()

            # compute residual
            self.build_residual()

            # compute norm of residual
            nf = np.linalg.norm(self.FUN)

            if self.solver_opts["monitor_convergence"]:
                print(f'norm(F) = {nf:.4e}')

            # check for convergence
            if nf < self.solver_opts["newton_tol"]:
                conv = True
                break

            # check for divergence
            if nf > 1e10:
                print('Newton iterations not converging')
                print(f'norm(F) = {nf:.4e}')
                break
            
            # builds the jacobian 
            self.build_jacobian()

            # update solution
            X -= np.linalg.solve(self.JAC, self.FUN)

        # print(f'{n} newton iterations needed')
        return X, conv
    

    def transient_response(self, opts = None):
        """
        Time steps the problem using the implicit Euler
        method. 

        Inputs
        ------
        opts: an optional argument that overwrites the default
        options for the newton solver


        Outputs
        -------
        sol: A Solution object that contains the solution components
        """

        # initalise solution object
        sol = Solution(self.pars)

        # extract time vector
        t = sol.t

        # overwrite default solver options if user provides
        # their own
        if opts != None:
            self.solver_opts = opts

        # initial condition
        self.u_old = np.zeros(self.N)
        self.lam_z_old = 1
        self.lam_r_old, self.lam_t_old = self.compute_stretches(self.u_old)
        
        # initial guess of solution
        X = self.set_initial_guess(sol)

        if self.loading == 'displacement':
            self.lam_z = self.pars.nondim["lam_z"]
        elif self.loading == 'force':
            self.F = self.pars.nondim["F"]
        else:
            print('ERROR: Unknown loading type')

        # begin time stepping
        for n in range(self.pars.computational["Nt"]):
            if self.solver_opts["monitor_convergence"]:
                print(f'----solving iteration {n}----')

            # assign step size
            self.dt = sol.dt[n]

            # solve for the next solution
            X, conv = self.newton_iterations(X)

            # check for convergence
            if not(conv):
                print(f'Newton iterations did not converge at step {n} (t = {t[n+1]:.2e})')
                sol.trim_solution(n)
                return sol

            # compute pressure
            if self.loading == 'displacement':
                self.compute_pressure()
                self.compute_force()

            # assign soln at previous time step
            self.u_old = X[0:self.N]
            self.lam_z_old = self.lam_z
            self.lam_t_old = self.lam_t
    
            # store soln
            sol.u[:, n+1] = self.u
            sol.p[:, n+1] = self.p
            sol.lam_z[n+1] = self.lam_z
            sol.F[n+1] = self.F
            sol.J[:, n+1] = self.J
            sol.phi[:, n+1] = 1 - (1 - self.pars.nondim["phi_0"]) / self.J

        return sol


    def numerical_jacobian(self):
        """
        Computes the Jacobian using finite differences
        """
        
        dx = 1e-5
        N = self.N

        if self.loading == 'displacement':
            self.JAC = np.zeros((N, N))
        else:
            self.JAC = np.zeros((2*N+1, 2*N+1))

        for i in range(N):
            self.u[i] += dx
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()
            self.build_residual()

            Fp = self.FUN.copy()

            self.u[i] -= 2 * dx
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()
            self.build_residual()

            Fm = self.FUN.copy()

            dF = (Fp - Fm) / 2 / dx
            self.JAC[:, i] = dF

            self.u[i] += dx
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()
            self.build_residual()

        if self.loading == 'force':
            for i in range(N):
                self.p[i] += dx
                self.build_residual()
                Fp = self.FUN.copy()

                self.p[i] -= 2*dx
                self.build_residual()
                Fm = self.FUN.copy()

                dF = (Fp - Fm) / 2 / dx
                self.JAC[:, self.ind_p[i]] = dF

                self.p[i] += dx
                self.build_residual()

            self.lam_z += dx
            self.compute_J()
            self.build_residual()
            Fp = self.FUN.copy()

            self.lam_z -= 2 * dx
            self.compute_J()
            self.build_residual()
            Fm = self.FUN.copy()

            dF = (Fp - Fm) / 2 / dx
            self.JAC[:, self.ind_l] = dF

            self.lam_z += dx
            self.compute_J()
            self.build_residual()

    def check_jacobian(self, X):
            """
            Compares the analytical and numerical Jacobians for
            debugging
            """

            # compute the numerical Jacobian using finite differences
            self.numerical_jacobian()
            J_n = self.JAC.copy()

            # compute the analytical Jacobian
            self.build_jacobian()
            J_a = self.JAC.copy()

            # compute the error row-by-row and print the result
            for i in range(len(X)):
                diff = np.linalg.norm(J_a[i,:] - J_n[i,:]) / np.linalg.norm(J_n[i,:], ord = np.inf)
                print(f'Row {i}: error = {diff:.4e}')         
        

