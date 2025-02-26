from .base_permeability import Permeability
from .base_permeability import sp

class HolmesMow(Permeability):
    """
    Defines the Holmes-Mow permeability.
    The constructor allows the user to pass custom values
    for the exponents.
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

        # Label
        self.label = 'Holmes-Mow'

        a = sp.Symbol('a')
        M = sp.Symbol('M')

        # Compute the solid fraction
        phi_s = 1 - self.phi

        # Introduce a small number to avoid dividing by zero if the initial
        # porosity is zero (as for a dry hydrogel)
        eps = 1e-10

        # Eulerian permeability
        self.k = self.k_0 * ((self.J - phi_s)/(self.phi_0 + eps))**a * sp.exp(1/2 * M * (self.J**2-1))
        
        # Build the model
        self.build()