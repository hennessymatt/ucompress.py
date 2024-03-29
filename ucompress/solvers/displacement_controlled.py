from .solution import Solution
from .experiment import Experiment, np
from scipy.optimize import root_scalar


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
                        



