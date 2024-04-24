import numpy as np


class DimensionalParameters():
    """
    A class to store dimensional parameter values for the problem.  
    The parameter values must be in SI units.

    The parameter attributes are split into four dictionaries:

    dimensional:    dimensional parameter values
    scalings:       scaling factors used to non-dimensionalise the
                    parameters
    nondim:         non-dimensionalised parameter values
    computational:  computational parameter values

    
    The dimensional/non-dimensional parameters must contain key/values for:

    R:      the initial radius of the sample
    G_m:    the shear modulus of the gel matrix
    k_0:    the initial permeability
    phi_0:  the initial porosity (fluid fraction)
    lam_z:  for displacement-controlled experiments, this is the imposed
            axial strain.  For force-controlled experiments, this is 
            the initial guess of the axial strain
    F:      for force-controlled experiments, this is the imposed force
            on the upper platten.  For displacement-controlled experiments,
            this value is not used and does not need to be assigned
    t_end:      the time of the final time step

    Of course, more parameters can be added to the above if needed by the 
    model.
    
    The computational parameters must contain key/values for:
    N:          the number of spatial grid points
    Nt:         the number of time steps to compute the solution at
    t_spacing:  either 'lin' or 'log'; determines whether to use linearly
                or logarithmically spaced time steps

    """
    def __init__(self):

        # empty dicts to store physical and computational parameters
        self.dimensional = {}
        self.nondim = {}
        self.computational = {}

    def compute_scaling_factors(self):
        """"
        Computes the scaling factors that are needed when non-dimensionalising 
        a set of parameter values. These scaling factors are then stored as an
        attribute in the form of a dictionary
        """

        space = self.dimensional["R"]
        stress = self.dimensional["G_m"]
        permeability = self.dimensional["k_0"]
        time = space**2 / stress / permeability
        force = stress * space**2
        
        self.scaling = {
            "space": space,
            "stress": stress,
            "permeability": permeability,
            "time": time,
            "force": force
        }

    def non_dimensionalise(self):
        """
        Carries out the non-dimensionalisation of all of the physical
        parameters as well as the final simulation time
        """
        self.nondim = {
            "R": self.dimensional["R"] / self.scaling["space"],
            "G_m": self.dimensional["G_m"] / self.scaling["stress"],
            "k_0": self.dimensional["k_0"] / self.scaling["permeability"],
            "phi_0": self.dimensional["phi_0"],
            "lam_z": self.dimensional["lam_z"],
            "F": self.dimensional["F"] / self.scaling["force"],
            "t_end": self.dimensional["t_end"] / self.scaling["time"]
        }


    def update(self, par, val):
        """
        Updates the scaling factors and non-dim parameters if
        the value of a dimensional parameter changes
        """

        if par in self.nondim:
            self.nondim[par] = val
        elif par in self.computational:
            self.computational[par] = val
        else:
            print('ERROR: parameter not found in dictionaries')

        self.compute_scaling_factors()
        self.non_dimensionalise()



class NonDimParameters():
    """
    A class to store non-dimension parameter values if these
    are known

    The parameter attributes are split into four dictionaries:

    dimensional:    dimensional parameter values
    computational:  computational parameter values

    The non-dimensional parameters must contain key/values for:

    R:      the initial radius of the sample
    G_m:    the shear modulus of the gel matrix
    k_0:    the initial permeability
    phi_0:  the initial porosity (fluid fraction)
    lam_z:  for displacement-controlled experiments, this is the imposed
            axial strain.  For force-controlled experiments, this is 
            the initial guess of the axial strain
    F:      for force-controlled experiments, this is the imposed force
            on the upper platten.  For displacement-controlled experiments,
            this value is not used and does not need to be assigned
    t_end:      the time of the final time step

    Of course, more parameters can be added to the above if needed by the 
    model.
    
    The computational parameters must contain key/values for:
    N:          the number of spatial grid points
    Nt:         the number of time steps to compute the solution at
    t_spacing:  either 'lin' or 'log'; determines whether to use linearly
                or logarithmically spaced time steps

    """

    def __init__(self):
        
        # empty dicts to store the parameters
        self.nondim = {}
        self.computational = {}