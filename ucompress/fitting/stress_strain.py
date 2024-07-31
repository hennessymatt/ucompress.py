from scipy.optimize import minimize
import numpy as np
from .chi_calculator import ChiCalculator

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

    def solve(self, fitting_params, fixed_hydration = False):
        """
        Carries out the minimisation using SciPy's solvers

        Inputs: 
        
        fitting_params = a dictionary where the keys are strings
        that describe the parameters to fit and the values are
        floats that are used to normalise the parameters so they
        all contribute to the optimisation equally

        Outputs:

        fitted_vals = a NumPy array with values of the fitted 
        """

        # Build the initial guess using the parameter values
        # already in the file
        N_pars = len(fitting_params)
        X = np.zeros(N_pars)

        for n, param in enumerate(fitting_params):
            # create the initial guess by normalising the parameters
            normalisation_factor = fitting_params[param]
            X[n] = self.pars.dimensional[param] / normalisation_factor

        # Solve the minimisation problem with SciPy
        result = minimize(
            lambda X: self.calculate_cost(X, fitting_params, fixed_hydration), 
            X, 
            method = 'BFGS',
            options = {"xrtol": 1e-3})

        print(result)

        # extract fitted values
        normalised_fitted_vals = result.x

        # un-normalised the fitted values
        fitted_vals = normalised_fitted_vals.copy()
        for n, param in enumerate(fitting_params):
            normalisation_factor = fitting_params[param]
            fitted_vals[n] = normalised_fitted_vals[n] * normalisation_factor            

        # print some info to the screen
        for val, param in zip(fitted_vals, fitting_params):
            print(f'{param} = {val:.2e}')


        return fitted_vals


    def calculate_cost(self, X, fitting_params, fixed_hydration):
        """
        Computes the objective function.  The compression of the
        material is assumed to be fast so that there is no loss
        of fluid.

        Inputs:

        X = array of values of the current fitting parameters
        
        fitting_params = a dictionary where the keys are strings
        that describe the parameters to fit and the values are
        floats that are used to normalise the parameters so they
        all contribute to the optimisation equally

        Outputs:

        cost = a float with the value of the cost/objective function

        """


        # Update the model with the new parameter values
        for value, param in zip(X, fitting_params):
            normalisation_factor = fitting_params[param]
            self.pars.update(param, value * normalisation_factor)
        self.model.assign(self.pars)

        # update hydration if required
        if fixed_hydration:
            chi_calc = ChiCalculator(self.model, self.pars)
            chi, beta_r, beta_z, phi_0 = chi_calc.solve(J_0 = self.pars.physical["J_h"])
            self.pars.update("chi", chi)
            self.pars.update("beta_r", beta_r)
            self.pars.update("beta_z", beta_z)
            self.pars.update("phi_0", phi_0)
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