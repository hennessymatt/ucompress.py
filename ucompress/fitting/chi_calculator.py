from scipy.optimize import root_scalar
from math import log

class ChiCalculator():
    """
    A class for calculating the value of \chi given
    a *single* value of the swelling ratio.
    """

    def __init__(self, model, pars):
        """
        Constructor, takes as arguments a model
        and parameter object
        """
        self.model = model
        self.pars = pars

    def solve(self, J_0):
        """
        Method for solving the hydration problem and
        computing \chi.  This method requires as
        input the swelling ratio J_0 and it returns
        the values of chi, the radial and axial stretches
        lam_r and lam_z, and the porosity phi
        """

        # first we set up the hydration problem and 
        # set the parameters for a dry gel
        self.pars.update("beta_r", 1)
        self.pars.update("beta_z", 1)
        self.pars.update("phi_0", 0)
        self.model.assign(self.pars)

        # calculate the porosity (fluid fraction)
        phi = 1 - 1 / J_0

        # solve the hydration problem
        sol_h = root_scalar(lambda x: self.hydration_fun(x, J_0), 
                            x0 = 0.2 * J_0, 
                            x1 = 0.9 * J_0, 
                            xtol = 1e-8,
                            bracket=(1.0, 1.1 * J_0)
                            )
        if not(sol_h.converged):
            raise Exception('Chi calculator did not converge')
        
        # extract the stretches
        lam_z = sol_h.root
        lam_r = (J_0 / lam_z)**(1/2)

        # compute the axial stress
        _, _, S_z = self.model.mechanics.eval_stress(lam_r, lam_r, lam_z)

        # compute the value of chi
        chi = -1 / (1 - phi)**2 * (S_z / self.pars.physical["G_T"] / lam_r**2 + log(phi) + 1 - phi)
        
        print(f'chi = {chi:.4f}')

        return chi, lam_r, lam_z, phi


    def hydration_fun(self, lam_z, J_0):
        """
        A helper method which defines the nonlinear system describing
        the hydrated equilibrium state.  This state can be calculated
        without knowledge of \chi provided that J_0 is known
        """

        lam_r = (J_0 / lam_z)**(1/2)
        S_r, _, S_z = self.model.mechanics.eval_stress(lam_r, lam_r, lam_z)

        F = S_r / lam_r / lam_z - S_z / lam_r**2
        return F
