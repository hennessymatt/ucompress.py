from .base_parameters import Parameters

class NeoHookean(Parameters):
    """
    An example parameter set for a neo-Hookean poroelastic material.
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
        Sets the parameter values in the model
        """

        # Define the dimensional physical parameters
        self.physical = {
            "R": 5e-3,          # Sample radius in reference configuration (m)

            "E_m": 50e3,        # Young's modulus of gel matrix (Pa)
            "nu_m": 0,          # Poisson's ratio of the matrix

            "k_0": 2e-13,       # initial hydraulic conductivity (m2 / Pa / s)
            "phi_0": 0.8,       # initial porosity (-)

            "lam_z": 0.5,       # axial strain (-)
            "beta_r": 1,        # radial pre-stretch (-)
            "beta_z": 1,        # axial pre-stretch (-)

            "F": -1,            # applied force (N)

            "t_start": 1e-1,    # first time point in simulation (s)
            "t_end": 1e4,       # final time point in simulation (s)
        }
        
        # Define the computational parameters
        self.computational = {
            "N": 40,            # Number of spatial grid points
            "Nt": 200,          # Number of time points
            "t_spacing": 'log'  # Spacing of time points
        }

###############################################################################

class Hydrogel(Parameters):
    """
    An example parameter set for a neo-Hookean hydrogel
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
        Sets the parameter values
        """

        # Define some material constants
        R_g = 8.314             # Univerisal gas constant (J/mol/K)
        V_m = 18e-6             # Molar volume of water (mol/m^3)
        T = 23 + 273            # Temperature (K)
        G_T = R_g * T / V_m     # Thermal stiffness (Pa)


        # Define the dimensional physical parameters
        self.physical = {
            "R": 5e-3,          # sample radius in reference configuration (m)

            "E_m": 50e3,        # Young's modulus of gel matrix (Pa)
            "nu_m": 0,          # Poisson's ratio of the matrix

            "k_0": 2e-13,       # initial hydraulic conductivity (m2 / Pa / s)

            "G_T": G_T,         # Thermal stiffness (Pa)
            "chi": 0.5,         # Flory interaction parameter (-)

            "beta_r": 1,        # radial pre-stretch (-)
            "beta_z": 1,        # axial pre-stretch (-)
            "phi_0": 0.0,       # initial porosity (-)

            "F": -1,            # applied force (N)
            "lam_z": 0.5,       # axial strain (-)

            "t_start": 1e-2,    # first time point in simulation (s)
            "t_end": 1e4,       # final time point in simulation (s)
        }
        
        # Define the computational parameters
        self.computational = {
            "N": 40,            # Number of spatial grid points
            "Nt": 100,          # Number of time points
            "t_spacing": 'log'  # Spacing of time points
        }     

    def non_dimensionalise(self):
        """
        We redefine (overload) the non_dimensionalise method because
        the thermal stiffness G_T in the dimensional parameter set
        needs to be non-dimensionalised
        """

        # non-dimensionalise all dimensional quantities
        self.physical["R"] /= self.scaling["space"]
        self.physical["E_m"] /= self.scaling["stress"]
        self.physical["G_T"] /= self.scaling["stress"]
        self.physical["k_0"] /= self.scaling["permeability"]
        self.physical["F"] /= self.scaling["force"]
        self.physical["t_start"] /= self.scaling["time"]
        self.physical["t_end"] /= self.scaling["time"]


###############################################################################

class FibreReinforced(Parameters):
    """
    An example parameter set for a fibre-reinforced 
    neo-Hookean poroelastic material (no osmotic effects)
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
        Sets the parameter values
        """

        # Define the dimensional physical parameters
        self.physical = {
            "R": 5e-3,          # sample radius in reference configuration (m)

            "E_m": 50e3,        # Young's modulus of gel matrix (Pa)
            "nu_m": 0,          # Poisson's ratio of the matrix
            
            "E_f": 50e6,        # Young's modulus of the fibres
            "Phi_f": 0.25,      # Nominal fibre fraction

            "k_0": 2e-13,       # initial hydraulic conductivity (m2 / Pa / s)

            "beta_r": 1,        # radial pre-stretch (-)
            "beta_z": 1,        # axial pre-stretch (-)
            "phi_0": 0.8,       # initial porosity (-)

            "F": -1,            # applied force (N)
            "lam_z": 0.5,       # axial strain (-)

            "t_start": 1e-2,    # first time point in simulation (s)
            "t_end": 1e4,       # final time point in simulation (s)
        }

        
        # Define the computational parameters
        self.computational = {
            "N": 40,            # Number of spatial grid points
            "Nt": 100,          # Number of time points
            "t_spacing": 'log'  # Spacing of time points
        }     

    def non_dimensionalise(self):
        """
        We redefine (overload) the non_dimensionalise method because
        the thermal stiffness G_T in the dimensional parameter set
        needs to be non-dimensionalised
        """

        # non-dimensionalise all dimensional quantities
        self.physical["R"] /= self.scaling["space"]
        self.physical["E_m"] /= self.scaling["stress"]
        self.physical["E_f"] /= self.scaling["stress"]
        self.physical["k_0"] /= self.scaling["permeability"]
        self.physical["F"] /= self.scaling["force"]
        self.physical["t_start"] /= self.scaling["time"]
        self.physical["t_end"] /= self.scaling["time"]

###############################################################################

class FibreRecruitment(Parameters):
    """
    An example parameter set for a fibre-reinforced 
    neo-Hookean poroelastic material (no osmotic effects)
    with fibre recruitment.  The parameters are such that
    the quartic recruitment distribution function is used.
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
        Sets the parameter values
        """

        # Define the dimensional physical parameters
        self.physical = {
            "R": 5e-3,          # sample radius in reference configuration (m)

            "E_m": 50e3,        # Young's modulus of gel matrix (Pa)
            "nu_m": 0,          # Poisson's ratio of the matrix
            
            "E_f": 50e6,        # Young's modulus of the fibres
            "Phi_f": 0.25,      # Nominal fibre fraction
            "lam_m": 2,         # Maximum recruitment stretch

            "k_0": 2e-13,       # initial hydraulic conductivity (m2 / Pa / s)

            "beta_r": 1,        # radial pre-stretch (-)
            "beta_z": 1,        # axial pre-stretch (-)
            "phi_0": 0.8,       # initial porosity (-)

            "F": -1,            # applied force (N)
            "lam_z": 0.5,       # axial strain (-)

            "t_start": 1e-2,    # first time point in simulation (s)
            "t_end": 1e4,       # final time point in simulation (s)
        }

        
        # Define the computational parameters
        self.computational = {
            "N": 40,            # Number of spatial grid points
            "Nt": 100,          # Number of time points
            "t_spacing": 'log'  # Spacing of time points
        }     

    def non_dimensionalise(self):
        """
        We redefine (overload) the non_dimensionalise method because
        the thermal stiffness G_T in the dimensional parameter set
        needs to be non-dimensionalised
        """

        # non-dimensionalise all dimensional quantities
        self.physical["R"] /= self.scaling["space"]
        self.physical["E_m"] /= self.scaling["stress"]
        self.physical["E_f"] /= self.scaling["stress"]
        self.physical["k_0"] /= self.scaling["permeability"]
        self.physical["F"] /= self.scaling["force"]
        self.physical["t_start"] /= self.scaling["time"]
        self.physical["t_end"] /= self.scaling["time"]
