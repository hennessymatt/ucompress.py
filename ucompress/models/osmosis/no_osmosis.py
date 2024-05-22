from .base_osmosis import OsmoticPressure

class NoOsmosis(OsmoticPressure):
    """
    Implementation of the osmotic pressure from the
    Flory-Huggins theory of solvent-polymer mixtures.
    The Flory interaction parameter is assumed to 
    be constant here.
    """

    def __init__(self):
        super().__init__()
        
        # Set the osmotic pressure to zero
        self.Pi = 0
            
        # Build the osmotic model
        self.build()