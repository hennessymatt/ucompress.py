from .base_permeability import Permeability

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