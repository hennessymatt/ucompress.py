from .base_osmosis import OsmoticPressure
from numpy import zeros, shape

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
        self.Pi = 0 * self.phi
            
        # Build the osmotic model
        self.build()

    def eval_osmotic_pressure(self, J):
        """
        Method that numerically evaluates Pi.

        Arguments
        --------
        J: the determinant (float or NumPy array)

        Returns
        -------
        Zero (of the same type as J)

        """

        return 0 * J
    
    def eval_osmotic_pressure_derivative(self, J):
        """
        Method that numerically evaluates the derivatives of Pi and returns
        
        Arguments
        --------
        J: the determinant (float or NumPy array)

        Returns
        -------
        Zero (of the same typeas J)

        """

        return 0 * J
        