import sympy as sp

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

    def build(self):
        """
        Builds the symbolic model.  For the permeability, this means
        computing the derivatives.  We use this method for consistency
        with the other parts of the model.
        """
        self.compute_derivatives()

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

        self.num_k = sp.lambdify(args, self.k.subs(pars.physical), translation)
        self.num_k_J = sp.lambdify(args, self.k_J.subs(pars.physical), translation)

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
        
