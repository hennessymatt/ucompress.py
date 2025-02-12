from ucompress.parameters.base_parameters import Parameters

class NeoHookean(Parameters):
    """
    An example non-dimensional parameter set for a neo-Hookean material
    """

    def __init__(self, nondim = False):

        """
        Constructor.  The keyword arg allows the user to specify whether the 
        parameters should be non-dimensionalised.  See the constructor of 
        the parent class for details
        """

        super().__init__(nondim = nondim)


    def set_parameters(self):

        """
        Physical parameters.
        """
        self.physical = {
            "R": 1,             # initial radius of sample
            "phi_0": 0.8,       # initial fluid fraction
            "beta_r": 1,        # initial radial pre-stretch
            "beta_z": 1,        # initial axial pre-stretch
            "k_0": 1,           # initial permeability
            "lam_z": 0.5,       # axial stretch
            "F": -5,            # force on the platten
            "E_m": 1,           # shear modulus of the matrix
            "nu_m": 0,          # Poisson's ratio of the matrix
            "t_start": 1e-4,    # start-time of simulation (needed due to log time stepping)
            "t_end": 10         # end-time of simulation
        }

        """
        computational parameters
        """
        self.computational = {
            "N": 40,            # number of spatial grid points
            "Nt": 100,          # number of time steps
            "t_spacing": 'log'  # lin or log spacing between time steps
        }