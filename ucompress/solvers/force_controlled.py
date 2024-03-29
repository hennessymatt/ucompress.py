from .solution import Solution
from .experiment import Experiment, np
from scipy.optimize import root_scalar


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
                        

