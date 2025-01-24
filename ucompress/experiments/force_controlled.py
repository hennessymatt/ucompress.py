from .solution import Solution
from .experiment import Experiment, np
from scipy.optimize import root_scalar, root

class ForceControlled(Experiment):
    """
    A class for force-controlled experiments
    """

    def __init__(self, model, pars):
        """
        Constructor, inherits the attribes from the Experiment
        class and adds a few more associated with specific 
        Jacobian entries
        """

        super().__init__(model, pars)
        self.loading = 'force'

        N = self.N

        # preallocate residual vectors
        self.FUN = np.zeros(2*N+1)
        self.F_p = np.zeros(N)

        # preallocate extra Jacobian entries
        self.J_up = np.zeros((N, N))
        self.J_ul = np.zeros((N, 1))

        self.J_pu = np.zeros((N, N))
        self.J_pl = np.zeros((N, 1))

        self.J_lu = np.zeros((1, N))
        self.J_lp = np.zeros((1, N))


    def initial_response(self, lam_z_0 = None):
        """
        Computes the initial response of the sample.

        Allows the user to provide an initial guess of the
        solution.  If none is provided, then the initial
        value of lam_z defined in the parameter object
        is used.
        """

        if lam_z_0 == None:
            lam_z_0 = self.pars.physical["lam_z"]
        

        # helper function for solving the nonlinea problem for
        # the initial response
        def fun(lam_z):
            self.lam_z = lam_z
            self.lam_r = np.array([1 / np.sqrt(self.lam_z)])
            self.lam_t = np.array([1 / np.sqrt(self.lam_z)])
            self.u = (1 / np.sqrt(self.lam_z) - 1) * self.r
            S_r, _, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
            self.p = 1 / np.sqrt(self.lam_z) * S_r
            self.compute_force()
            
            return self.pars.physical["F"] - self.F
        
        # solve the nonlinear scalar equation for the axial stretch
        sol = root_scalar(fun, 
                          method = 'secant', 
                          x0 = lam_z_0, 
                          x1 = lam_z_0 * 1.01
                          )
        
        # check if the solver converged
        if sol.converged == True:
            fun(sol.root)
        else:
            print(sol)
            raise Exception('ERROR: solver for initial response did not converge')
        
        # compute fluid load fraction
        self.compute_fluid_load_fraction()

        # create a solution object and store the solution
        sol = Solution(self.pars, 0)
        sol.u = self.u
        sol.lam_z = self.lam_z
        sol.p = self.p
        sol.F = self.F
        sol.fluid_load_fraction = self.fluid_load_fraction

        return sol

    def steady_response(self, lam_r_0 = None, lam_z_0 = None):
        """
        Computes the steady response of the sample.

        Allows the user to provide an initial guess of the
        solution (radial and axial stretch).  If none are
        provided, then default values are used
        """

        if lam_r_0 == None:
            lam_r_0 = 1.1
        if lam_z_0 == None:
            lam_z_0 = self.pars.physical["lam_z"]

        # helper function for solving the nonlinear problem for
        # the steady response
        def fun(X):
            self.lam_r = X[0]
            self.lam_t = X[0]
            self.lam_z = X[1]
            self.p = self.osmosis.eval_osmotic_pressure(self.lam_r**2 * self.lam_z)
            S_r, _, S_z = self.mech.eval_stress(self.lam_r, self.lam_r, self.lam_z)
            self.compute_force()
            
            return np.array([
                S_r - self.lam_r * self.lam_z * self.p,
                self.pars.physical["F"] - self.F
            ])
        
        # solve the nonlinear scalar equation for the axial stretch
        steady_sol = root(fun, x0 = np.array([lam_r_0, lam_z_0]))
    
        # check if the solver converged
        if steady_sol.success == True:
            fun(steady_sol.x)
        else:
            print(steady_sol)
            raise Exception('ERROR: solver for steady response did not converge')
        
        # compute fluid load fraction
        self.compute_fluid_load_fraction()

        # create a solution object and store the solution
        sol = Solution(self.pars, 0)
        sol.u = steady_sol.x[0] * sol.r
        sol.lam_z = steady_sol.x[1]
        sol.p = self.p
        sol.F = self.F
        sol.fluid_load_fraction = self.fluid_load_fraction

        return sol


    def set_initial_guess(self, sol):
        """
        Sets the initial guess of the solution to
        the small-time (instantaneous response) solution.

        Inputs:
        sol - A Solution object which contains time points

        Outputs:
        X - the initial guess of the solution
        """

        R = self.pars.physical["R"]
        t_pe = R**2 / self.pars.physical["k_0"] / self.pars.physical["E_m"]

        # compute the initial response
        self.initial_response()

        Pi = self.osmosis.eval_osmotic_pressure(self.lam_r**2 * self.lam_z)

        # assume a boundary-layer-type solution for the pressure
        self.p = (self.p[0] - Pi) * (1 - np.exp(-(1-self.r / R) / (sol.t[1]/t_pe)**(1/2))) + Pi

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

        # Evaluate stresses, permeability, and osmotic pressure
        S_r, S_t, S_z = self.mech.eval_stress(self.lam_r, self.lam_t, self.lam_z)
        k = self.perm.eval_permeability(self.J)
        Pi = self.osmosis.eval_osmotic_pressure(self.J)

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
        self.F_u[1:-1] = self.r[1:-1] / 2 * self.dLdt[1:-1] - k[1:-1] / self.lam_r[1:-1] * (
            self.div_S - self.lam_t[1:-1] * self.lam_z * (self.D[1:-1,:] @ Pi)
        )

        # BCs for u
        self.F_u[0] = self.u[0]
        self.F_u[-1] = S_r[-1] - self.lam_t[-1] * self.lam_z * Pi[-1]

        #----------------------------------------------------
        # pressure
        #----------------------------------------------------
        self.F_p[:-1] = self.D[:-1,:] @ (self.p - Pi) - self.r[:-1] * self.lam_r[:-1]**2 / 2 / k[:-1] / self.J[:-1] * self.dLdt[:-1]
        self.F_p[-1] = self.p[-1] - Pi[-1]

        #----------------------------------------------------
        # axial stretch
        #----------------------------------------------------
        self.F_l = 2 * np.pi * np.sum(self.w * (S_z - self.lam_r * self.lam_t * self.p) * self.r) - self.pars.physical["F"]

        #----------------------------------------------------
        # build the global residual vector
        #----------------------------------------------------
        self.FUN[self.ind_u] = self.F_u
        self.FUN[self.ind_p] = self.F_p
        self.FUN[self.ind_l] = self.F_l


    def analytical_jacobian(self):
        """
        Updates the entries in the Jacobian for the stress balance
        """

        # compute derivatives of the elastic stresses
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
        self.J_uu[1:-1, :] = np.diag(self.lam_z * self.r / self.dt)[1:-1,:] @ np.diag(self.lam_t) @ self.lam_t_u

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

        self.J_ul[1:-1,0] = (
            self.r[1:-1] / 2 / self.dt * self.lam_t[1:-1]**2 - 
            k_J[1:-1] * self.J_l[1:-1] / self.lam_r[1:-1] * (self.div_S - self.lam_t[1:-1] * self.lam_z * (self.D[1:-1,:] @ Pi)) - 
            k[1:-1] / self.lam_r[1:-1] * (
                self.D[1:-1,:] @ S_r_z + (S_r_z - S_t_z)[1:-1] / self.r[1:-1] - 
                self.lam_t[1:-1] * (self.D[1:-1,:] @ Pi) - self.lam_t[1:-1] * self.lam_z * (self.D[1:-1,:] @ (Pi_J * self.J_l))
                )
        )

        # boundary conditions for u
        self.J_uu[-1, :] = (
            S_r_r[-1,:] @ self.lam_r_u + S_r_t[-1,:] @ self.lam_t_u 
            - self.lam_z * (Pi[-1] * self.lam_t_u[-1,:] + self.lam_t[-1] * Pi_J[-1] * self.J_u[-1,:])
        )

        self.J_ul[-1, 0] = (
            S_r_z[-1] - 
            self.lam_t[-1] * Pi[-1] - 
            self.lam_t[-1] * self.lam_z * (self.J_l[-1] * Pi_J[-1])
        )
        
        #----------------------------------------------------
        # pressure
        #----------------------------------------------------
        self.J_pu[:-1,:] = -(
            np.diag(self.r * self.lam_r * self.dLdt / k / self.J)[:-1,:] @ self.lam_r_u - 
            np.diag(self.r * self.lam_r**2 * k_J * self.dLdt / 2 / k**2 / self.J)[:-1,:] @ self.J_u - 
            np.diag(self.r * self.lam_r**2 * self.dLdt / 2 / k / self.J**2)[:-1,:] @ self.J_u + 
            np.diag(self.r * self.lam_r**2 * self.lam_t * self.lam_z / k / self.J / self.dt)[:-1,:] @ self.lam_t_u
        ) - self.D[:-1,:] @ np.diag(Pi_J) @ self.J_u

        self.J_pl[:-1, 0] = -(
            -self.r * self.lam_r**2 * k_J * self.J_l / 2 / k**2 / self.J * self.dLdt - 
            self.r * self.lam_r**2 * self.J_l / 2 / k / self.J**2 * self.dLdt + 
            self.r * self.lam_r**2 / 2 / k / self.J / self.dt * self.lam_t**2
        )[:-1] - self.D[:-1,:] @ (Pi_J * self.J_l)

        # boundary condition for p
        self.J_pu[-1,:] = -Pi_J[-1] * self.J_u[-1,:]
        self.J_pl[-1,0] = -Pi_J[-1] * self.J_l[-1]

        #----------------------------------------------------
        # axial stretch
        #----------------------------------------------------
        self.J_lu = 2 * np.pi * np.dot(
            self.w, 
            np.diag(S_z_r * self.r) @ self.lam_r_u + 
            np.diag(S_z_t * self.r) @ self.lam_t_u -
            np.diag(self.lam_t * self.p * self.r) @ self.lam_r_u -
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
                        

