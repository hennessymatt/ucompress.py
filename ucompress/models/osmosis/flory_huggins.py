from .base_osmosis import sp
from .base_osmosis import OsmoticPressure

class FloryHuggins(OsmoticPressure):
    """
    Implementation of the osmotic pressure from the
    Flory-Huggins theory of solvent-polymer mixtures.
    The Flory interaction parameter is assumed to 
    be constant here.
    """

    def __init__(self):
        super().__init__()

        chi = sp.Symbol('chi')  # Flory interaction parameter
        G_T = sp.Symbol('G_T')  # Thermal stiffness = R_g * T / V_w 

        self.Pi = -G_T * (
            sp.log(self.phi) + chi * (1 - self.phi)**2 + (1 - self.phi)
        )
            
        # Build the osmotic model
        self.build()