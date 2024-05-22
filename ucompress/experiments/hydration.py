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


    def steady_response(self):
        """
        Computes the steady-state response to hydration
        """

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
            
        # Solve the problem 
        sol = root(fun, [1.5, 1.5])

        if sol.success:
            print('Hydration solver converged')

            lam_r = sol.x[0]
            lam_z = sol.x[1]

            J = lam_r**2 * lam_z

            print(f'Volumetric expansion due to hydration: {J:.2f}')
            print(f'Radial stretch: {lam_r:.2f}')
            print(f'Axial stretch: {lam_z: .2f}')
            

        else:
            raise Exception('Hydration solver did not converge')