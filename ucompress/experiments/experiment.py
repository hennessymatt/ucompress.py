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
        self.mech = model.mechanics
        self.perm = model.permeability
        self.osmosis = model.osmosis
        self.pars = pars

        self.preallocate()

        # Set the default solver options
        self.solver_opts = {
            "jacobian": "analytical", # use analytical Jacobian for Newton iterations
            "monitor_convergence": False, # monitor convergence of newton iterations
            "newton_max_iterations": 10, # maximum number of newton iterations
            "abs_tol": 1e-8, # absolute error (of residual) convergence tolerance
            "rel_tol": 1e-8, # relative error (of residual) convergence tolerance
            "div_tol": 1e20 # max absolute error (divergence criteria)
        }

    def preallocate(self):
        """
        Preallocates variables that are common to both solvers
        """

        # number of grid points
        N = self.pars.computational["N"]
        self.N = N

        # sample radius
        R = self.pars.physical["R"]

        # indices
        self.ind_u = np.arange(N)
        self.ind_p = np.arange(N, 2*N)
        self.ind_l = 2*N

        # build operators
        D, y = cheb(self.N)
        self.D = 2 * D / R
        self.r = R * (y + 1) / 2

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
        self.lam_t_u[0,:] = self.D[0, :]
        self.lam_t_u[1:,1:] = np.diag(1 / self.r[1:])

        # weights for the trapezoidal rule
        self.w = np.zeros(N)
        dr = np.diff(self.r)
        self.w[0] = dr[0] / 2
        self.w[-1] = dr[-1] / 2
        self.w[1:-1] = (dr[1:] + dr[:-1]) / 2



    def compute_stretches(self, u):
        """
        Computes the radial and orthoradial stretches.

        Arguments
        ---------
        u: the radial displacement

        Returns
        -------
        lam_r: the radial stretch
        lam_t: the orthoradial stretch

        """

        # Compute radial stretch
        lam_r = 1 + self.D @ u

        # Compute orthoradial stretch.  The first component is obtained
        # using L'Hopital's rule  
        lam_t = np.r_[1 + self.D[0,:] @ u, 1 + u[1:] / self.r[1:]]

        # Return the stretches
        return lam_r, lam_t


    def compute_J(self):
        """
        Computes the Jacobian (J = det(F)) and its derivatives with respect
        to the displacement and axial stretch
        """

        # Compute J
        self.J = self.lam_r * self.lam_t * self.lam_z
        
        # Compute derivative of J wrt displacement u
        self.J_u = self.lam_z * (np.diag(self.lam_t) @ self.lam_r_u + np.diag(self.lam_r) @ self.lam_t_u)
        
        # Compute derivative of J wrt axial stretch lambda_z
        self.J_l = self.lam_r * self.lam_t


    def compute_pressure(self):
        """
        Computes the pressure if the deformation is known
        """

        # Computer permeability and osmotic pressure
        k = self.perm.eval_permeability(self.J)
        Pi = self.osmosis.eval_osmotic_pressure(self.J)

        # Construct RHS of linear ODE
        rhs = self.r * self.lam_r**2 / 2 / k / self.J / self.dt * (
            self.lam_t**2 * self.lam_z - self.lam_t_old**2 * self.lam_z_old
            ) + self.D @ Pi
        
        # Impose the boundary condition p = Pi at r = R
        rhs[-1] = Pi[-1]

        # Solve
        self.p = np.linalg.solve(self.J_pp, rhs)


    def compute_force(self):
        """
        Computes the force on the platten if the deformation
        and pressure are known using the trapezoidal rule
        """

        # Evaluate the mechanical stress
        _, _, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)

        # Compute the force
        self.F = 2 * np.pi * np.sum(self.w * 
                                    (S_z - self.p * self.lam_r * self.lam_t) 
                                    * self.r)

    def compute_fluid_load_fraction(self):
        """
        Computes the fluid load fraction, mathematically defined
        as the area integral of mu * J / lam_z divided by the
        force F
        """

        # Calculate the chem potential
        Pi = self.osmosis.eval_osmotic_pressure(self.lam_r**2 * self.lam_z)
        mu = self.p - Pi

        # Calculate the FLF
        self.fluid_load_fraction = -2 * np.pi * np.sum(
            self.w * (mu * self.lam_r * self.lam_t) * self.r
            ) / self.F

    def newton_iterations(self, X):
        """
        Implementation of Newton's method.

        Arguments
        ----------
        X:  The initial guess of the solution

        Returns
        --------
        X:      The final solution approximation
        conv:   Convergence flag
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

            # store initial residual norm for rel_tol convergence
            if n == 0:
                nf_0 = nf

            if self.solver_opts["monitor_convergence"]:
                print(f'norm(F) = {nf:.4e}')

            # check for convergence
            if nf < self.solver_opts["abs_tol"] or nf / nf_0 < self.solver_opts["rel_tol"]:
                conv = True
                break

            # check for divergence
            if nf > self.solver_opts["div_tol"]:
                print('Newton iterations not converging')
                print(f'norm(F) = {nf:.4e}')
                break
            
            # builds the jacobian 
            self.build_jacobian()
            # self.check_jacobian(X)
            # conv = False
            # break

            # update solution
            X -= np.linalg.solve(self.JAC, self.FUN)

            # increment counter
            self.total_newton_iterations += 1

        # print(f'{n} newton iterations needed')
        return X, conv
    

    def transient_response(self, solver_opts = "default"):
        """
        Time steps the problem using the implicit Euler
        method. 

        Inputs
        ------
        solver_opts: an optional argument that overwrites the default
        options for the newton solver


        Outputs
        -------
        sol: A Solution object that contains the solution components
        """

        # overwrite the default solver options
        if solver_opts != "default":
            self.solver_opts = solver_opts
        

        # initalise solution object
        sol = Solution(self.pars)

        # extract time vector
        t = sol.t

        # Set up the Jacobian
        if self.solver_opts["jacobian"] == "analytical":
            self.build_jacobian = self.analytical_jacobian
        elif self.solver_opts["jacobian"] == "numerical":
            self.build_jacobian = self.numerical_jacobian
        else:
            raise Exception('Unknown Jacobian type!')


        # initial condition
        self.u_old = np.zeros(self.N)
        self.lam_z_old = 1
        self.lam_r_old, self.lam_t_old = self.compute_stretches(self.u_old)
        
        # initial guess of solution
        X = self.set_initial_guess(sol)

        if self.loading == 'displacement':
            self.lam_z = self.pars.physical["lam_z"]
        elif self.loading == 'force':
            self.F = self.pars.physical["F"]
        else:
            print('ERROR: Unknown loading type')

        # begin time stepping
        self.total_newton_iterations = 0
        print('--------------------------------')
        print('Transient step')
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
                sol.trim_solution(n+1)
                return sol

            # compute pressure
            if self.loading == 'displacement':
                self.compute_pressure()
                self.compute_force()

            # compute fluid load fraction
            self.compute_fluid_load_fraction()

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
            sol.phi[:, n+1] = 1 - (1 - self.pars.physical["phi_0"]) / self.J
            sol.fluid_load_fraction[n+1] = self.fluid_load_fraction

        print('Solver converged')
        mean_newton_iterations = self.total_newton_iterations / self.pars.computational["Nt"]
        print(f'Average number of Newton iterations per time step: {mean_newton_iterations:.1f}')

        return sol


    def numerical_jacobian(self):
        """
        Computes the Jacobian using finite differences
        """
        
        # Increment for finite differences
        dx = 1e-5
        N = self.N

        # Pre-allocate Jacobian
        if self.loading == 'displacement':
            self.JAC = np.zeros((N, N))
        else:
            self.JAC = np.zeros((2*N+1, 2*N+1))

        # Loop over u components
        for i in range(N):

            # Forwards approximation
            self.u[i] += dx
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()
            self.build_residual()

            Fp = self.FUN.copy()

            # Backwards difference
            self.u[i] -= 2 * dx
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()
            self.build_residual()

            Fm = self.FUN.copy()

            # Central difference
            dF = (Fp - Fm) / 2 / dx
            self.JAC[:, i] = dF

            # Reset the value of u
            self.u[i] += dx
            self.lam_r, self.lam_t = self.compute_stretches(self.u)
            self.compute_J()
            self.build_residual()

        # Add extra Jacobian entries if solving for pressure and lam_z
        if self.loading == 'force':
            for i in range(N):

                # Forwards
                self.p[i] += dx
                self.build_residual()
                Fp = self.FUN.copy()

                # Backwards
                self.p[i] -= 2*dx
                self.build_residual()
                Fm = self.FUN.copy()

                # Central difference
                dF = (Fp - Fm) / 2 / dx
                self.JAC[:, self.ind_p[i]] = dF

                # Reset
                self.p[i] += dx
                self.build_residual()

            # Forwards
            self.lam_z += dx
            self.compute_J()
            self.build_residual()
            Fp = self.FUN.copy()

            # Backwards
            self.lam_z -= 2 * dx
            self.compute_J()
            self.build_residual()
            Fm = self.FUN.copy()

            # Central differencing
            dF = (Fp - Fm) / 2 / dx
            self.JAC[:, self.ind_l] = dF

            # Reset
            self.lam_z += dx
            self.compute_J()
            self.build_residual()

