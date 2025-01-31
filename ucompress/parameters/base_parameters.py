class Parameters():
    """
    A class to store parameter values for the problem.  

    The parameter attributes are split into two main dictionaries:

    physical:       parameter values associate with the physical system,
                    e.g. the material or the experiment
                    
    computational:  computational parameter values assigned by the user.

    
    The dictionary of physical parameters *must* contain key/values for:

    R:          Radius of the sample in reference configuration
    E_m:        Young's modulus of the gel matrix
    nu_m:       Poisson's ratio of the gel matrix
    k_0:        Initial permeability in reference configuration
    phi_0:      Initial porosity (fluid fraction) in ref configuration
    lam_z:      For displacement-controlled experiments, this is the imposed
                axial stretch.  For force-controlled experiments, this is 
                the initial guess of the axial stretch
    beta_r:     Radial pre-stretch (e.g. from hydration)
    beta_z:     Axial pre-stretch (e.g. from hydration)
    F:          For force-controlled experiments, this is the imposed force
                on the upper platten.  For displacement-controlled experiments,
                this value is not used and does not need to be assigned
    t_start:    the first time step of the simulation
    t_end:      the time of the final time step

    Of course, more parameters can be added to the above if needed by the 
    model.
        
    The dictionary of computational parameters must contain key/values for:

    N:          the number of spatial grid points
    Nt:         the number of time steps to compute the solution at
    t_spacing:  either 'lin' or 'log'; determines whether to use linearly
                or logarithmically spaced time steps

    If you are working with dimensional variables, then this class contains
    methods for converting these to non-dimensional variables, which is
    better for the solver.  There is also a method for updating non-dim
    variables of a dimensional variable changes.

    """
    def __init__(self, nondim = False):
        """
        Constructor.  The keyword arg allows the user to specify whether the 
        parameters should be non-dimensionalised 
        """

        # Save the non-dim flag
        self.nondim = nondim

        # Create the parameter dicts by calling the method from the
        # child class
        self.set_parameters()

        # Non-dimensionalise if user chooses this option
        if nondim:

            # create a new dict to store the dimensional parameter values
            self.dimensional = self.physical.copy()

            # compute the scaling factors needed to non-dim the problem
            self.compute_scaling_factors()

            # non-dim the parameters in the physical dict (overwritten)
            self.non_dimensionalise()

    def set_parameters(self):
        """
        Template method for setting the relevant parameter dicts. In
        practice, this method is overloaded by child classes that 
        contain parameter values
        """

        # Stores physical parameter values
        self.physical = {}

        # Stores computational parameter values
        self.computational = {}

    def compute_scaling_factors(self):
        """"
        Computes the scaling factors that are needed when non-dimensionalising 
        a set of parameter values. These scaling factors are then stored as an
        attribute in the form of a dictionary.
        """

        space = self.dimensional["R"]
        stress = self.dimensional["E_m"]
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
        parameters as well as the start/end simulation time.  In its 
        current form, this method only applies to simple neo-Hookean
        materials.  This method will therefore have to be overloaded
        if using a more complex model with extra parameters.
        """

        # non-dimensionalise all dimensional quantities
        self.physical["R"] /= self.scaling["space"]
        self.physical["E_m"] /= self.scaling["stress"]
        self.physical["k_0"] /= self.scaling["permeability"]
        self.physical["F"] /= self.scaling["force"]
        self.physical["t_start"] /= self.scaling["time"]
        self.physical["t_end"] /= self.scaling["time"]


    def update(self, par = None, val = None):
        """
        Updates the scaling factors and non-dim parameters if
        the value of a dimensional parameter changes.  If 
        dimensional parameters are being used, then this
        just updates the entries in the physical/computational
        dicts
        """

        """
        If using non-dim parameters, copy the dimensional dict
        into the physical dict so it can be overwritten with
        the new non-dim values
        """
        if self.nondim:
            self.physical = self.dimensional.copy()

        """
        Update the parameter values in the relevant dict
        """
        if par != None and val != None:
            if par in self.physical:
                self.physical[par] = val
            elif par in self.computational:
                self.computational[par] = val
            else:
                raise Exception('ERROR: parameter not found in dictionaries')

        """
        Non-dim again if required
        """
        if self.nondim:
            # update the dimensional dict
            self.dimensional = self.physical.copy()

            # recompute scaling factor
            self.compute_scaling_factors()

            # non-dimensionalise
            self.non_dimensionalise()


    def __str__(self):
        """
        Controls how Parameter objects are printed with Python's
        print function
        """

        str = (
            'Dimensional parameter values' +
            '\n' + 
            '---------------------------------------' +
            '\n'
        )

        # Extract the dimensional parameter set
        if self.nondim:
            dimensional = self.dimensional
        else:
            dimensional = self.physical

        for k in dimensional:
            str += f'{k} = {dimensional[k]:.2e}\n'

        # Print the non-dim parameters if they are being used
        if self.nondim:
            str += (
                '\n' +
                'Non-dimensional parameter values' +
                '\n' + 
                '-----------------------------------------' +
                '\n'
            )

            for k in self.physical:
                str += f'{k} = {self.physical[k]:.2e}\n'

        # Prints the computational parameters
        str += (
            '\n' +
            'Computational parameter values' +
            '\n' + 
            '-----------------------------------------' +
            '\n'
        )

        for k in self.computational:
            str += f'{k} = {self.computational[k]   }\n'


        return str