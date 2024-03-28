import sympy as sp
import numpy as np


class Permeability():
    """
    Superclass for all permeability models
    """
    def __init__(self):
        """
        Constructor for the superclass
        """
        # define attributes as SymPy symbols
        self.k_0 = sp.Symbol('k_0')
        self.phi_0 = sp.Symbol('phi_0')
        self.J = sp.Symbol('J')

        # define Eulerian porosity
        self.phi = 1 - (1 - self.phi_0) / self.J

    def compute_derivatives(self):
        """
        Method to compute derivatives of the permeability wrt state variables
        """
        self.k_J = sp.diff(self.k, self.J)

    def lambdify(self, pars):
        """
        Converts SymPy expressions into NumPy expressions
        """
        args = [self.J]
        translation = "numpy"

        self.num_k = sp.lambdify(args, self.k.subs(pars), translation)
        self.num_k_J = sp.lambdify(args, self.k_J.subs(pars), translation)

    def eval_permeability(self, J):
        """
        Method that numerically evaluates K and returns a NumPy array
        """
        return self.num_k(J)
    
    def eval_permeability_derivative(self, J):
        """
        Method that numerically evaluates the derivatives of K and returns
        NumPy arrays
        """
        return self.num_k_J(J)
        

class Constant(Permeability):
    """
    A class for a constant Eulerian permeability.
    Here we overload the evaluation method to 
    ensure that evaluation of k and its derivatives
    returns a NumPy array with the same length of u
    """

    def __init__(self, pars):
        """
        Constructor
        """
        super().__init__()

        # Eulerian permeability
        self.k = self.k_0
        
        # compute derivatives of the Lagrangian permeability
        self.compute_derivatives()

        # convert SymPy to NumPy
        self.lambdify(pars)

    def eval_permeability(self, J):
        """
        Method that numerically evaluates K and returns a NumPy array
        """
        return self.num_k(J) * np.ones(len(J))
    
    def eval_permeability_derivative(self, J):
        """
        Method that numerically evaluates the derivatives of K and returns
        NumPy arrays
        """
        return self.num_k_J(J) * np.ones(len(J))


class KozenyCarman(Permeability):
    """
    Defines the Kozeny-Carman permeability.
    The constructor allows the user to pass custom values
    for the exponents.
    """

    def __init__(self, pars):
        """
        Constructor
        """
        super().__init__()

        # Eulerian permeability
        self.k = self.k_0 * (1 - self.phi_0)**2 / self.phi_0**3 * self.phi**3 / (1 - self.phi)**2
        
        # compute derivatives of the Lagrangian permeability
        self.compute_derivatives()

        # convert SymPy to NumPy
        self.lambdify(pars)