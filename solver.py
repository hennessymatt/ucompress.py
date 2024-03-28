import numpy as np
from scipy.optimize import root_scalar
from cheb import *

class Solution():
    """
    Class for storing the outputs of the solvers
    """
    def __init__(self, t, u, p, lam_z, F):

        self.t = t
        self.u = u
        self.p = p
        self.lam_z = lam_z
        self.F = F

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

    def preallocate(self):
        """
        Preallocates variables that are common to both solvers
        """

        # number of grid points
        N = self.pars.N
        self.N = N

        # indices
        self.ind_u = np.arange(N)
        self.ind_p = np.arange(N, 2*N)
        self.ind_l = 2*N
        self.ind_F = 2*N + 1

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


    def newton_iterations(self, X, opts):
        """
        Implementation of Newton's method
        """
        
        conv = False
        for n in range(self.pars.newton_max_iterations):

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

            if opts["monitor_convergence"]:
                print(f'norm(F) = {nf:.4e}')

            # check for convergence
            if nf < self.pars.newton_conv_tol:
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
    

    def solve(self, opts = None):
        """
        Time steps the problem using the implicit Euler
        method. 

        Inputs
        ------
        X: A NumPy array containing the initial guess of 
        the solution


        Outputs
        -------
        X_all: A NumPy array containing the solution (u, p, lambda_z) at
        each space and time point
        """

        # initial condition
        self.u_old = np.zeros(self.N)
        self.lam_z_old = 1
        self.lam_r_old, self.lam_t_old = self.compute_stretches(self.u_old)
        
        # initial guess of solution
        X = self.set_initial_guess()

        if self.loading == 'displacement':
            self.lam_z = self.pars.lam_z
        elif self.loading == 'force':
            self.F = self.pars.lam_z
        else:
            print('ERROR: Unknown loading type')

        # preallocate
        X_all = np.zeros((2 * self.N+2, self.pars.Nt+1))
        X_all[self.ind_l, 0] = 1

        # begin time stepping
        for n in range(self.pars.Nt):
            if opts["monitor_convergence"]:
                print(f'----solving iteration {n}----')

            # assign step size
            self.dt = self.pars.dt[n]

            # solve for the next solution
            X, conv = self.newton_iterations(X, opts)

            # check for convergence
            if not(conv):
                print(f'Newton iterations did not converge at step {n} (t = {self.pars.t[n+1]:.2e})')
                return X_all[:,:n]

            # compute pressure
            if self.loading == 'displacement':
                self.compute_pressure()
                self.compute_force()

            # assign soln at previous time step
            self.u_old = X[0:self.N]
            self.lam_z_old = self.lam_z
            self.lam_t_old = self.lam_t
    
            # store soln
            X_all[self.ind_u, n+1] = self.u
            X_all[self.ind_p, n+1] = self.p
            X_all[self.ind_l, n+1] = self.lam_z
            X_all[self.ind_F, n+1] = self.F

        sol = Solution(self.pars.t, 
                X_all[self.ind_u,:], 
                X_all[self.ind_p,:], 
                X_all[self.ind_l,:],
                X_all[self.ind_F,:]
                )

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
        


class DisplacementControlled(Experiment):

    def __init__(self, pars, model):

        super().__init__(pars, model)
        self.loading = 'displacement'

        # preallocate some more variables
        self.FUN = np.zeros(self.N)


    def initial_response(self):
        """
        Computes the initial response of the sample
        """

        self.lam_z = self.pars.lam_z
        self.lam_r = np.array([1 / np.sqrt(self.lam_z)])
        self.lam_t = np.array([1 / np.sqrt(self.lam_z)])
        self.u = (1 / np.sqrt(self.lam_z) - 1) * self.r
        S_r, _, _ = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
        self.p = 1 / np.sqrt(self.lam_z) * S_r
        self.compute_force()

        return Solution(0, self.u, self.p, self.lam_z, self.F)

    def set_initial_guess(self):
        """
        Sets the initial guess of the solution to
        the initial response
        """

        self.initial_response()
        
        # set the initial guess of the solution
        X = np.r_[
            self.u,
            ]
                
        return X


    def build_residual(self):
        """
        Builds the residual
        """ 

        # Evaluate stresses and permeability
        S_r, S_t, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
        k = self.perm.eval_permeability(self.J)

        # compute div of elastic stress tensor
        self.div_S = self.D[1:-1,:] @ S_r + (S_r[1:-1] - S_t[1:-1]) / self.r[1:-1]
        
        # bulk eqn for u
        self.F_u[1:-1] = self.r[1:-1] / 2 / self.dt * (
            self.lam_t[1:-1]**2 * self.lam_z - self.lam_t_old[1:-1]**2 * self.lam_z_old
            ) - k[1:-1] / self.lam_r[1:-1] * self.div_S

        # BCs for u
        self.F_u[0] = self.u[0]
        self.F_u[-1] = S_r[-1]

        #----------------------------------------------------
        # build the global residual vector
        #----------------------------------------------------
        self.FUN = self.F_u


    def build_jacobian(self):
        """
        Updates the entries in the Jacobian for the stress balance
        """

        (S_r_r, S_r_t, S_r_z, 
        S_t_r, S_t_t, S_t_z,
        S_z_r, S_z_t, S_z_z) = self.mech.eval_stress_derivatives(self.lam_r, self.lam_t, self.lam_z)

        k = self.perm.eval_permeability(self.J)
        k_J = self.perm.eval_permeability_derivative(self.J)

        #----------------------------------------------------
        # displacement
        #----------------------------------------------------

        # diff d/dt stuff
        self.J_uu[1:-1, :] = np.diag(self.lam_z * self.r / self.dt)[1:-1,:] @ np.diag(self.lam_t) @ self.lam_t_u

        # diff effective perm
        self.J_uu[1:-1, :] -= np.diag(self.div_S) @ (
            np.diag(k_J / self.lam_r)[1:-1,:] @ self.J_u - 
            np.diag(k[1:-1] / self.lam_r[1:-1]**2) @ self.lam_r_u[1:-1,:]
        )

        # diff div(S)
        self.J_uu[1:-1, :] -= np.diag(k[1:-1] / self.lam_r[1:-1]) @ (
            self.D[1:-1, :] @ (S_r_r @ self.lam_r_u + S_r_t @ self.lam_t_u)
            + np.diag(1 / self.r[1:-1]) @ (
                S_r_r[1:-1,:] @ self.lam_r_u + S_r_t[1:-1,:] @ self.lam_t_u - 
                S_t_r[1:-1,:] @ self.lam_r_u - S_t_t[1:-1,:] @ self.lam_t_u)
            )

        # boundary conditions for u
        self.J_uu[-1, :] = S_r_r[-1, :] @ self.lam_r_u + S_r_t[-1,:] @ self.lam_t_u

        #----------------------------------------------------
        # build the global block Jacobian
        #----------------------------------------------------
        self.JAC = self.J_uu
                        




class ForceControlled(Experiment):

    def __init__(self, pars, model):

        super().__init__(pars, model)
        self.loading = 'force'

        N = self.N

        # residual vectors
        self.FUN = np.zeros(2*N+1)
        self.F_p = np.zeros(N)

        # define extra Jacobian entries
        self.J_up = np.zeros((N, N))
        self.J_ul = np.zeros((N, 1))

        self.J_pu = np.zeros((N, N))
        self.J_pl = np.zeros((N, 1))

        self.J_lu = np.zeros((1, N))
        self.J_lp = np.zeros((1, N))


    def initial_response(self):
        """
        Computes the initial response of the sample
        """

        def fun(lam_z):
            self.lam_z = lam_z
            self.lam_r = np.array([1 / np.sqrt(self.lam_z)])
            self.lam_t = np.array([1 / np.sqrt(self.lam_z)])
            self.u = (1 / np.sqrt(self.lam_z) - 1) * self.r
            S_r, _, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
            self.p = 1 / np.sqrt(self.lam_z) * S_r
            self.compute_force()

            return self.pars.F - self.F
        
        sol = root_scalar(fun, method = 'secant', x0=self.pars.lam_z, x1 = self.pars.lam_z * 1.01)
        if sol.converged == True:
            fun(sol.root)
        else:
            print('ERROR: solver for initial response did not converge')

        return Solution(0, self.u, self.p[0], self.lam_z, self.F)


    def set_initial_guess(self):
        """
        Sets the initial guess of the solution
        """

        self.initial_response()
        self.p = self.p[0] * (1 - np.exp(-(1-self.r) / self.pars.t[1]**(1/2)))

        # set the initial guess of the solution
        X = np.r_[
            self.u,
            self.p,
            self.lam_z
            ]
                
        return X


    def build_residual(self):
        """
        Builds the residual
        """ 

        # Evaluate stresses and permeability
        S_r, S_t, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
        k = self.perm.eval_permeability(self.J)

        #----------------------------------------------------
        # displacement
        #----------------------------------------------------

        # compute div of elastic stress tensor
        self.div_S = self.D[1:-1,:] @ S_r + (S_r[1:-1] - S_t[1:-1]) / self.r[1:-1]

        # compute time derivative of L = lam_t**2 * lam_z
        self.dLdt = 1 / self.dt * (
            self.lam_t**2 * self.lam_z - self.lam_t_old**2 * self.lam_z_old
            )

        # bulk eqn for u
        self.F_u[1:-1] = self.r[1:-1] / 2 * self.dLdt[1:-1] - k[1:-1] / self.lam_r[1:-1] * self.div_S

        # BCs for u
        self.F_u[0] = self.u[0]
        self.F_u[-1] = S_r[-1]

        #----------------------------------------------------
        # pressure
        #----------------------------------------------------
        self.F_p[:-1] = self.D[:-1,:] @ self.p - self.r[:-1] * self.lam_r[:-1]**2 / 2 / k[:-1] / self.J[:-1] * self.dLdt[:-1]
        # self.F_p[:-1] = self.D[:-1,:] @ self.p - self.r[:-1] / 2 / k[:-1] * self.dLdt[:-1]
        self.F_p[-1] = self.p[-1]

        #----------------------------------------------------
        # axial stretch
        #----------------------------------------------------
        self.F_l = 2 * np.pi * np.sum(self.w * (S_z - self.lam_r * self.lam_t * self.p) * self.r) - self.pars.F

        #----------------------------------------------------
        # build the global residual vector
        #----------------------------------------------------
        self.FUN[self.ind_u] = self.F_u
        self.FUN[self.ind_p] = self.F_p
        self.FUN[self.ind_l] = self.F_l


    def build_jacobian(self):
        """
        Updates the entries in the Jacobian for the stress balance
        """
        (S_r_r, S_r_t, S_r_z, 
        S_t_r, S_t_t, S_t_z,
        S_z_r, S_z_t, S_z_z) = self.mech.eval_stress_derivatives(self.lam_r, self.lam_t, self.lam_z)

        k = self.perm.eval_permeability(self.J)
        k_J = self.perm.eval_permeability_derivative(self.J)

        #----------------------------------------------------
        # displacement
        #----------------------------------------------------

        # diff d/dt stuff
        self.J_uu[1:-1, :] = np.diag(self.lam_z * self.r / self.dt)[1:-1,:] @ np.diag(self.lam_t) @ self.lam_t_u

        # diff effective perm
        self.J_uu[1:-1, :] -= np.diag(self.div_S) @ (
            np.diag(k_J / self.lam_r)[1:-1,:] @ self.J_u - 
            np.diag(k[1:-1] / self.lam_r[1:-1]**2) @ self.lam_r_u[1:-1,:]
        )

        # diff div(S)
        self.J_uu[1:-1, :] -= np.diag(k[1:-1] / self.lam_r[1:-1]) @ (
            self.D[1:-1, :] @ (S_r_r @ self.lam_r_u + S_r_t @ self.lam_t_u)
            + np.diag(1 / self.r[1:-1]) @ (
                S_r_r[1:-1,:] @ self.lam_r_u + S_r_t[1:-1,:] @ self.lam_t_u - 
                S_t_r[1:-1,:] @ self.lam_r_u - S_t_t[1:-1,:] @ self.lam_t_u)
            )

        self.J_ul[1:-1,0] = (
            self.r[1:-1] / 2 / self.dt * self.lam_t[1:-1]**2 - 
            k_J[1:-1] * self.J_l[1:-1] / self.lam_r[1:-1] * self.div_S - 
            k[1:-1] / self.lam_r[1:-1] * (self.D[1:-1,:] @ S_r_z + (S_r_z - S_t_z)[1:-1] / self.r[1:-1])
        )

        # boundary conditions for u
        self.J_uu[-1, :] = S_r_r[-1,:] @ self.lam_r_u + S_r_t[-1,:] @ self.lam_t_u
        self.J_ul[-1,0] = S_r_z[-1]

        #----------------------------------------------------
        # pressure
        #----------------------------------------------------
        self.J_pu[:-1,:] = -(
            np.diag(self.r * self.lam_r * self.dLdt / k / self.J)[:-1,:] @ self.lam_r_u - 
            np.diag(self.r * self.lam_r**2 * k_J * self.dLdt / 2 / k**2 / self.J)[:-1,:] @ self.J_u - 
            np.diag(self.r * self.lam_r**2 * self.dLdt / 2 / k / self.J**2)[:-1,:] @ self.J_u + 
            np.diag(self.r * self.lam_r**2 * self.lam_t * self.lam_z / k / self.J / self.dt)[:-1,:] @ self.lam_t_u
        )

        self.J_pl[:-1, 0] = -(
            -self.r * self.lam_r**2 * k_J * self.J_l / 2 / k**2 / self.J * self.dLdt - 
            self.r * self.lam_r**2 * self.J_l / 2 / k / self.J**2 * self.dLdt + 
            self.r * self.lam_r**2 / 2 / k / self.J / self.dt * self.lam_t**2
        )[:-1]

        # self.J_pu[:-1,:] = -(
        #     -np.diag(self.r * k_J * self.dLdt / 2 / k**2)[:-1,:] @ self.J_u +
        #     np.diag(self.r * self.lam_t * self.lam_z / k / self.dt)[:-1,:] @ self.lam_t_u
        # )

        # self.J_pl[:-1, 0] = -(
        #     -self.r * k_J * self.J_l / 2 / k**2 * self.dLdt +
        #     self.r / 2 / k / self.dt * self.lam_t**2
        # )[:-1]


        #----------------------------------------------------
        # axial stretch
        #----------------------------------------------------
        self.J_lu = -2 * np.pi * np.dot(
            self.w, 
            np.diag(self.lam_t * self.p * self.r) @ self.lam_r_u + 
            np.diag(self.lam_r * self.p * self.r) @ self.lam_t_u
        )
        self.J_lp = -2 * np.pi * self.w * self.lam_r * self.lam_t * self.r
        self.J_ll = 2 * np.pi * np.sum(self.w * S_z_z * self.r)

        #----------------------------------------------------
        # build the global block Jacobian
        #----------------------------------------------------
        self.JAC = np.block([[self.J_uu, self.J_up, self.J_ul], 
                             [self.J_pu, self.J_pp, self.J_pl], 
                             [self.J_lu, self.J_lp, self.J_ll]
                             ])
                        

