import sympy as sp

class OsmoticPressure():
    """
    Superclass for all models of the osmotic pressure
    """
    def __init__(self):
        """
        Constructor for the superclass
        """
        # define attributes as SymPy symbols
        self.phi_0 = sp.Symbol('phi_0')
        self.J = sp.Symbol('J')

        # define Eulerian porosity
        self.phi = 1 - (1 - self.phi_0) / self.J

    def build(self):
        """
        Builds the symbolic model.  For the osmotic pressure, this means
        computing the derivatives.  We use this method for consistency
        with the other parts of the model.
        """
        self.compute_derivatives()

    def compute_derivatives(self):
        """
        Method to compute derivatives of the osmotic pressire wrt 
        state variables
        """
        self.Pi_J = sp.diff(self.Pi, self.J)

    def lambdify(self, pars):
        """
        Converts SymPy expressions into NumPy expressions
        """
        args = [self.J]
        translation = "numpy"

        self.num_Pi = sp.lambdify(args, self.Pi.subs(pars.physical), translation)
        self.num_Pi_J = sp.lambdify(args, self.Pi_J.subs(pars.physical), translation)

    def eval_osmotic_pressure(self, J):
        """
        Method that numerically evaluates Pi and returns a NumPy array
        """
        return self.num_Pi(J)
    
    def eval_osmotic_pressure_derivative(self, J):
        """
        Method that numerically evaluates the derivatives of Pi and returns
        NumPy arrays
        """
        return self.num_Pi_J(J)
        
