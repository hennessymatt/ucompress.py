from scipy.optimize import root

class Hydration():
    """
    A class for free-swelling hydration experiments
    """

    def __init__(self, model, parameters):
        """
        The constructor requires model and parameter objects
        """
        self.mech = model.mechanics
        self.osmosis = model.osmosis
        self.pars = parameters


    def steady_response(self, lam_r = None, lam_z = None):
        """
        Computes the steady-state response to hydration.  The user
        can provide initial guesses for the radial and axial
        stretches if they want.  If not, they default to
        prescribed values. 

        Returns:
        lam_r - the radial stretch
        lam_z - the axial stretch
        phi_0 - the porosity
        """

        """
        Check that the hydration stretches are set to one
        """
        if abs(self.pars.physical["beta_r"] - 1) > 1e-6:
            raise Exception('beta_r > 1 in hydration solver')
        if abs(self.pars.physical["beta_z"] - 1) > 1e-6:
            raise Exception('beta_z > 1 in hydration solver')
        
        """
        Check that the initial porosity is zero (dry gel)
        """
        if self.pars.physical["phi_0"] > 1e-3:
            raise Exception('phi_0 > 0 in hydration solver')

        # Set the initial guess if not passed by the user
        if lam_r == None:
            lam_r = 1.1
        
        if lam_z == None:
            lam_z = 1.1


        def fun(X):
            """
            A helper function that defines the final hydration
            state. 
            """
            lam_r = X[0]
            lam_z = X[1]
            J = lam_r**2 * lam_z

            Pi = self.osmosis.eval_osmotic_pressure(J)

            S_r, _, S_z = self.mech.eval_stress(lam_r, lam_r, lam_z)
            
            # The residual is that the radial and axial forces are zero
            return [
                S_r - lam_r * lam_z * Pi,
                S_z - lam_r**2 * Pi
            ]
            
        print('----------------------------------------')
        print('Hydration step')

        # Solve the problem 
        sol = root(fun, [lam_r, lam_z], tol = 1e-8)

        if sol.success:
            print('Solver converged')

            lam_r = sol.x[0]
            lam_z = sol.x[1]

            J = lam_r**2 * lam_z
            phi_0 = 1 - 1 / J

            print(f'Volumetric expansion due to hydration: {J:.4f}')
            print(f'Fluid fraction in hydrated state: {phi_0:.4f}')
            print(f'Radial stretch: {lam_r:.4f}')
            print(f'Axial stretch: {lam_z: .4f}')
            
            return lam_r, lam_z, phi_0

        else:
            raise Exception('ERROR: Solver did not converge')