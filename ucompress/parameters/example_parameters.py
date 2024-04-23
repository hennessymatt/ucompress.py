from .base_parameters import BaseParameters

class NeoHookean(BaseParameters):
    """
    An example parameter set for a neo-Hookean material
    """
    def __init__(self):

        super().__init__()

        """
        Physical parameters.
        """
        self.physical = {
            "R": 1,         # initial radius of sample
            "phi_0": 0.8,   # initial fluid fraction
            "k_0": 1,       # initial permeability
            "lam_z": 0.5,   # axial stretch
            "F": -5,        # force on the platten
            "G_m": 1,       # shear modulus of the matrix
        }

        """
        computational parameters
        """
        self.computational = {
            "N": 40,            # number of spatial grid points
            "Nt": 100,          # number of time steps
            "t_end": 10,        # end-time of simulation
            "t_spacing": 'log'  # lin or log spacing between time steps
        }


class FibreReinforcedNH(BaseParameters):
    """
    An example parameter set for a fibre-reinforced neo-Hookean material
    """
    def __init__(self):

        super().__init__()

        """
        Physical parameters. 
        """

        self.physical = {
            "R": 1,         # initial radius of sample
            "phi_0": 0.8,   # initial fluid fraction
            "k_0": 1,       # initial permeability
            "lam_z": 0.5,   # axial stretch
            "F": -5,        # force on the platten
            "G_m": 1,       # shear modulus of the matrix
            "G_f": 100,     # shear modulus of the fibres
            "alpha_f": 0.5  # volume fraction of fibres
        }

        """
        computational parameters
        """
        self.computational = {
            "N": 40,            # number of spatial grid points
            "Nt": 100,          # number of time steps
            "t_end": 10,        # end-time of simulation
            "t_spacing": 'log'  # lin or log spacing between time steps
        }