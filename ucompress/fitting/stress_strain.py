from scipy.optimize import minimize
import numpy as np

class StressStrain():
    """
    Class for fitting model parameters to stress-strain data
    """

    def __init__(self, strain_data, stress_data, model, pars):
        """
        Constructor: stress and strain data must be positive.
        The stress data must have units of Pa
        """

        self.stretch_data = self.strain_to_stretch(strain_data)
        self.stress_data = stress_data
        self.N_data = len(strain_data)

        self.model = model
        self.pars = pars

    def strain_to_stretch(self, strain_data):
        """
        Converts strain data (mm/mm) into stretch (lambda_z)
        data
        """
        return 1 - strain_data

    def solve(self, fitting_params):
        """
        Carries out the minimisation using SciPy's solvers

        Inputs: 
        
        fitting_params = a tuple of strings corresponding to
        the fitting parameters

        Outputs:

        fitted_vals = a NumPy array with values of the fitted 
        """

        # Build the initial guess using the parameter values
        # already in the file
        N_pars = len(fitting_params)
        X = np.zeros(N_pars)

        for n, param in enumerate(fitting_params):
            X[n] = self.pars.physical[param]

        # Solve the minimisation problem with SciPy
        result = minimize(lambda X: self.calculate_cost(X, fitting_params), X, tol=1e-2)

        print(result)

        # extract fitted values
        fitted_vals = result.x

        # print some info to the screen
        for val, param in zip(fitted_vals, fitting_params):
            print(f'{param} = {val:.2e}')


        return fitted_vals


    def calculate_cost(self, X, fitting_params):
        """
        Computes the objective function.  The compression of the
        material is assumed to be fast so that there is no loss
        of fluid.

        Inputs:

        X = array of values of the current fitting parameters
        
        fitting_params = a tuple of strings corresponding to
        the fitting parameters

        Outputs:

        cost = a float with the value of the cost/objective function

        """

        # Update the model with the new parameter values
        for value, param in zip(X, fitting_params):
            self.pars.update(param, value)
        
        self.model.assign(self.pars)

        # load the data points to evaluate the stretch at
        lam_z = self.stretch_data
        
        # compute radial stretch
        lam_r = 1 / np.sqrt(lam_z)

        # compute stresses and pressure
        S_r, _, S_z = self.model.mechanics.eval_stress(lam_r, lam_r, lam_z)
        p = lam_r * S_r

        # Calculate the total PK1 stress
        S_z_T = S_z - lam_r**2 * p

        # Compute the cost as the RMSE
        cost = np.sqrt(np.mean((-self.pars.scaling["stress"] * S_z_T - self.stress_data)**2))

        print(X, cost)

        return cost