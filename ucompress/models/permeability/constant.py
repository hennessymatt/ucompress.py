from .base_permeability import Permeability
from numpy import ones

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
        return self.num_k(J) * ones(len(J))
    
    def eval_permeability_derivative(self, J):
        """
        Method that numerically evaluates the derivatives of K and returns
        NumPy arrays
        """
        return self.num_k_J(J) * ones(len(J))