import numpy as np


class BaseParameters():
    """
    A class to store the parameter values for the problem.  The parameters
    are split into physical parameters and computational parameters.  Both
    are stored as dictionary attributes.

    The physical parameters must contain key/values for:
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


    The computational parameters must contain key/values for:
    N:          the number of spatial grid points
    Nt:         the number of time steps to compute the solution at
    t_end:      the time of the final time step
    t_spacing:  either 'lin' or 'log'; determines whether to use linearly
                or logarithmically spaced time steps

    """
    def __init__(self):

        # empty dicts to store physical and computational parameters
        self.physical = {}
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
            "time": time,
            "force": force,
            "permeability": permeability
        }

    def update(self, par, val):
        """
        Updates the scaling factors and non-dim parameters if
        the value of a dimensional parameter changes
        """

        if par in self.physical:
            self.physical[par] = val
        elif par in self.computational:
            self.computational[par] = val
        else:
            print('ERROR: parameter not found in dictionaries')

        self.compute_scaling_factors()
        self.non_dimensionalise()