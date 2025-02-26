from .base_permeability import Permeability

class KozenyCarman(Permeability):
    """
    Defines the Kozeny-Carman permeability.
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

        # Label
        self.label = 'Kozeny-Carmen'

        # Introduce a small number to avoid dividing by zero if the initial
        # porosity is zero (as for a dry hydrogel)
        eps = 1e-10

        # Eulerian permeability
        self.k = self.k_0 * (1 - self.phi_0)**2 / (self.phi_0 + eps)**3 * self.phi**3 / (1 - self.phi)**2
        
        # Build the model
        self.build()