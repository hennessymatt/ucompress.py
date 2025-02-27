from .osmosis.no_osmosis import NoOsmosis

class Poroelastic():
    """
    A class for a poroelastic material without any osmotic
    effects, e.g. fluid transport due to pressure gradients only
    """

    def __init__(self, mechanics, permeability, parameters):
        """
        Constructor requires a mechanics, permeability,
        and parameters object.  The permeability object can
        be set to None if it is not required in the experiment.
        """

        # Label
        self.label = 'Poroelastic'


        self.mechanics = mechanics
        self.permeability = permeability
        self.osmosis = NoOsmosis()

        """
        Evaluates the symbolic model using the parameter values
        """
        self.assign(parameters)


    def assign(self, parameters):
        """
        A method to assign the symbolic model using the parameter values.
        This triggers the conversion from a SymPy symbolic model into a
        collection of NumPy functions
        """

        self.mechanics.lambdify(parameters)
        if self.permeability is not None:
            self.permeability.lambdify(parameters)

class Hydrogel():
    """
    A class for hydrogels, i.e. poroelastic materials with osmotic
    effects.  The flow is driven by gradients in composition
    and pressure. The permeability object can
    be set to None if it is not required in the experiment.
    """

    def __init__(self, mechanics, permeability, osmosis, parameters):
        """Constructor requires a mechanics, permeability, osmosis,
        and parameters object"""

        # Label
        self.label = 'Hydrogel'

        self.mechanics = mechanics
        self.permeability = permeability
        self.osmosis = osmosis

        """
        Evaluates the symbolic model using the parameter values
        """
        self.assign(parameters)


    def assign(self, parameters):
        """
        A method to assign the symbolic model using the parameter values.
        This triggers the conversion from a SymPy symbolic model into a
        collection of NumPy functions
        """

        self.mechanics.lambdify(parameters)
        if self.permeability is not None:
            self.permeability.lambdify(parameters)
        self.osmosis.lambdify(parameters)