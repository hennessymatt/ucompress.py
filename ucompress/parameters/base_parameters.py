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
    lam_z:  for displacement-controlled experiments, this is the impose
            axial strain.  For force-controlled experiments, this is 
            the initial guess of the axial strain
    F:      the force-controlled experiments, this is the imposed force
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

        