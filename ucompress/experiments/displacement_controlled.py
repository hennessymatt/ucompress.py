from .solution import Solution
from .experiment import Experiment, np
from scipy.optimize import root

class DisplacementControlled(Experiment):

    def __init__(self, model, pars):

        super().__init__(model, pars)
        self.loading = 'displacement'

        # preallocate some more variables
        self.FUN = np.zeros(self.N)


    def initial_response(self):
        """
        Computes the initial response of the sample
        """

        self.lam_z = self.pars.physical["lam_z"]
        self.lam_r = np.array([1 / np.sqrt(self.lam_z)])
        self.lam_t = np.array([1 / np.sqrt(self.lam_z)])
        self.u = (1 / np.sqrt(self.lam_z) - 1) * self.r
        S_r, _, _ = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
        self.p = 1 / np.sqrt(self.lam_z) * S_r
        self.compute_force()
        self.compute_fluid_load_fraction()

        # create a solution object for the initial response
        sol = Solution(self.pars, 0)
        sol.u = self.u
        sol.lam_z = self.lam_z
        sol.p = self.p
        sol.F = self.F
        sol.fluid_load_fraction = self.fluid_load_fraction

        return sol
    
    def steady_response(self):
        """
        Computes the steady-state response
        """

        # Set the axial stretch
        self.lam_z = self.pars.physical["lam_z"]

        # helper function for solving the nonlinear problem for
        # the steady response
        def fun(X):
            self.lam_r = X
            self.lam_t = X
            self.p = self.osmosis.eval_osmotic_pressure(self.lam_r**2 * self.lam_z)
            S_r, _, _ = self.mech.eval_stress(self.lam_r, self.lam_r, self.lam_z)
            self.compute_force()
            
            return S_r - self.lam_r * self.lam_z * self.p
            
        
        # solve the nonlinear scalar equation for the axial stretch
        steady_sol = root(fun, x0 = np.array([1.1]))
    
        # check if the solver converged
        if steady_sol.success == True:
            fun(steady_sol.x)
        else:
            print('ERROR: solver for initial response did not converge')
            return
        
        # compute fluid load fraction
        self.compute_fluid_load_fraction()

        # create a solution object and store the solution
        sol = Solution(self.pars, 0)
        sol.u = steady_sol.x[0] * sol.r
        sol.p = self.p
        sol.F = self.F
        sol.fluid_load_fraction = self.fluid_load_fraction

        return sol


    def set_initial_guess(self, sol = None):
        """
        Sets the initial guess of the solution to
        the small-time (instantaneous response) solution.

        The argument sol is not used but is needed for consistency
        with the force-controlled case.

        Outputs:
        X - the initial guess of the solution
        """

        # compute the initial response
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
        Pi = self.osmosis.eval_osmotic_pressure(self.J)

        # compute div of elastic stress tensor
        self.div_S = self.D[1:-1,:] @ S_r + (S_r[1:-1] - S_t[1:-1]) / self.r[1:-1]
        
        # bulk eqn for u
        self.F_u[1:-1] = self.r[1:-1] / 2 / self.dt * (
            self.lam_t[1:-1]**2 * self.lam_z - self.lam_t_old[1:-1]**2 * self.lam_z_old
            ) - k[1:-1] / self.lam_r[1:-1] * (
                self.div_S - self.lam_t[1:-1] * self.lam_z * (self.D[1:-1,:] @ Pi)
            )


        # BCs for u
        self.F_u[0] = self.u[0]
        self.F_u[-1] = S_r[-1] - self.lam_t[-1] * self.lam_z * Pi[-1]

        #----------------------------------------------------
        # build the global residual vector
        #----------------------------------------------------
        self.FUN = self.F_u


    def analytical_jacobian(self):
        """
        Updates the entries in the Jacobian for the stress balance
        """

        # compute the stress derivatives
        (S_r_r, S_r_t, S_r_z, 
        S_t_r, S_t_t, S_t_z,
        S_z_r, S_z_t, S_z_z) = self.mech.eval_stress_derivatives(self.lam_r, self.lam_t, self.lam_z)

        # compute the permeability and its derivative wrt J
        k = self.perm.eval_permeability(self.J)
        k_J = self.perm.eval_permeability_derivative(self.J)

        # compute the osmotic pressure and its derivative
        Pi = self.osmosis.eval_osmotic_pressure(self.J)
        Pi_J = self.osmosis.eval_osmotic_pressure_derivative(self.J)

        #----------------------------------------------------
        # displacement
        #----------------------------------------------------

        # diff d/dt stuff
        self.J_uu[1:-1, :] = np.diag(self.lam_t * self.lam_z * self.r / self.dt)[1:-1,:] @ self.lam_t_u

        # diff effective perm
        self.J_uu[1:-1, :] -= np.diag(self.div_S - self.lam_t[1:-1] * self.lam_z * (self.D[1:-1,:] @ Pi)) @ (
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

        # diff lam_t * lam_z * d(Pi)/dr terms
        self.J_uu[1:-1, :] += np.diag(self.lam_z * k[1:-1] / self.lam_r[1:-1]) @ (
            np.diag(self.D[1:-1,:] @ Pi) @ self.lam_t_u[1:-1,:] + 
            np.diag(self.lam_t[1:-1]) @ self.D[1:-1,:] @ (np.diag(Pi_J) @ self.J_u)
        )

        # boundary conditions for u
        self.J_uu[-1, :] = (
            S_r_r[-1,:] @ self.lam_r_u + S_r_t[-1,:] @ self.lam_t_u 
            - self.lam_z * (Pi[-1] * self.lam_t_u[-1,:] + self.lam_t[-1] * Pi_J[-1] * self.J_u[-1,:])
        )


        #----------------------------------------------------
        # build the global block Jacobian
        #----------------------------------------------------
        self.JAC = self.J_uu