from .base_parameters import BaseParameters

class NeoHookean(BaseParameters):
    """
    An example non-dimensional parameter set for a neo-Hookean material
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


class DimensionalNeoHookean(BaseParameters):
    """
    An example non-dimensional parameter set for a neo-Hookean material.
    Here the non-dim parameters are converted from a dimensional parameter
    set
    """

    def __init__(self):
        
        super().__init__()

        """
        Define the dimensional parameters in SI units
        """
        self.dimensional = {
            "R": 5e-3,        # initial radius (m)
            "G_m": 50e3,      # stiffness of gel matrix (Pa)
            "k_0": 2e-13,     # initial hydraulic conductivity (m2 / Pa / s)
            "phi_0": 0.8,     # initial porosity (-)
            "lam_z": 0.5,     # axial strain (-)
            "F": -1,          # applied force (N)
            "t_end": 1e4,     # final time (s)
        }

        """
        Define the computational parameters that are independent of the
        non-dimensionalisation
        """
        self.computational = {
            "N": 40,
            "Nt": 100,
            "t_spacing": 'log'
        }

        """
        Compute the scaling factors that are needed to non-dimensionalise
        the paramters
        """
        self.compute_scaling_factors()
    
        """
        Computes dicts of non-dim parameters
        """
        self.non_dimensionalise()


    def non_dimensionalise(self):
        """
        Carries out the non-dimensionalisation of all of the physical
        parameters as well as the final simulation time
        """
        self.physical = {
            "R": self.dimensional["R"] / self.scaling["space"],
            "G_m": self.dimensional["G_m"] / self.scaling["stress"],
            "k_0": self.dimensional["k_0"] / self.scaling["permeability"],
            "phi_0": self.dimensional["phi_0"],
            "lam_z": self.dimensional["lam_z"],
            "F": self.dimensional["F"] / self.scaling["force"]
        }

        self.computational["t_end"] = self.dimensional["t_end"] / self.scaling["time"]
        
    

class FibreReinforcedNH(BaseParameters):
    """
    An example non-dimensional parameter set for a fibre-reinforced 
    neo-Hookean material
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