import numpy as np
import sympy as sp

class Mechanics():
    """
    This is a generic (super)class that is used to store
    information about the mechanical response of the material.
    This class itself does not define the mechanics response;
    rather, it provides common methods for e.g. computing
    stress derivatives from subclasses in which the precise
    mechanics are defined.
    """

    def __init__(self):
        """
        Constructor, defines the stretches, pressure,
        and some invariants
        """

        # Pre-stretches due to hydration
        self.beta_r, self.beta_z = sp.symbols('beta_r beta_z')

        # Stretches from the hydrated state
        self.lam_r = sp.Symbol('lam_r')
        self.lam_t = sp.Symbol('lam_t')
        self.lam_z = sp.Symbol('lam_z')

        # Total stretches
        self.Lam_r = self.beta_r * self.lam_r
        self.Lam_t = self.beta_r * self.lam_t
        self.Lam_z = self.beta_z * self.lam_z

        # Determinants
        self.J_0 = self.beta_r**2 * self.beta_z
        self.J = self.lam_r * self.lam_t * self.lam_z
        self.J_t = self.J * self.J_0

        # Invariants
        self.I_1 = self.Lam_r**2 + self.Lam_t**2 + self.Lam_z**2
        self.I_1_x = self.Lam_r**2 + self.Lam_t**2

        """
        Create an empty dictionary to store any functions that need to be
        converted from SymPy into SciPy.  The user can add entries to this
        entries when defining their model
        """
        self.conversion_dict = {}


    def stress_derivatives(self):
        """
        Method that computes the derivatives of the stresses
        wrt the stretches and pressure using SymPy
        """
        self.sig_r_r = sp.diff(self.sig_r, self.lam_r)
        self.sig_r_t = sp.diff(self.sig_r, self.lam_t)
        self.sig_r_z = sp.diff(self.sig_r, self.lam_z)

        self.sig_t_r = sp.diff(self.sig_t, self.lam_r)
        self.sig_t_t = sp.diff(self.sig_t, self.lam_t)
        self.sig_t_z = sp.diff(self.sig_t, self.lam_z)

        self.sig_z_r = sp.diff(self.sig_z, self.lam_r)
        self.sig_z_t = sp.diff(self.sig_z, self.lam_t)
        self.sig_z_z = sp.diff(self.sig_z, self.lam_z)
          
    def lambdify(self, pars):
        """
        Turns the SymPy expressions for the stresses and
        their derivatives into fast NumPy functions

        We need to create a copy of the physical parameter dictionary
        without lam_z.  This avoids a bug where the stresses are
        always evaluated using the value of lam_z assigned in the
        parameter dictionary (which is fine for displacement-controlled
        loading but not force-controlled loading)
        """

        # Copy the physical parameters dict and remove lam_z
        pars = pars.physical.copy()
        pars.pop("lam_z")

        # Define the arguments of the NumPy function
        args = [self.lam_r, self.lam_t, self.lam_z]

        # Instructions on how to translate SymPy into Numpy
        translation = [self.conversion_dict, 'numpy']

        # Lambdify the code
        self.S_r = sp.lambdify(args, self.sig_r.subs(pars), translation)
        self.S_t = sp.lambdify(args, self.sig_t.subs(pars), translation)
        self.S_z = sp.lambdify(args, self.sig_z.subs(pars), translation)

        self.S_r_r = sp.lambdify(args, self.sig_r_r.subs(pars), translation)
        self.S_r_t = sp.lambdify(args, self.sig_r_t.subs(pars), translation)
        self.S_r_z = sp.lambdify(args, self.sig_r_z.subs(pars), translation)

        self.S_t_r = sp.lambdify(args, self.sig_t_r.subs(pars), translation)
        self.S_t_t = sp.lambdify(args, self.sig_t_t.subs(pars), translation)
        self.S_t_z = sp.lambdify(args, self.sig_t_z.subs(pars), translation)

        self.S_z_r = sp.lambdify(args, self.sig_z_r.subs(pars), translation)
        self.S_z_t = sp.lambdify(args, self.sig_z_t.subs(pars), translation)
        self.S_z_z = sp.lambdify(args, self.sig_z_z.subs(pars), translation)


    def eval_stress(self, lam_r, lam_t, lam_z):
        """
        Numerically evaluates the stresses and returns them
        """
        return (
            self.S_r(lam_r, lam_t, lam_z), 
            self.S_t(lam_r, lam_t, lam_z), 
            self.S_z(lam_r, lam_t, lam_z)
        )
    
    def eval_stress_derivatives(self, lam_r, lam_t, lam_z):
        """
        Numerically evaluates the derivatives of the stresses 
        and returns them
        """

        N = len(lam_r)
        O = np.ones(N)

        return (
            np.diag(self.S_r_r(lam_r, lam_t, lam_z)),
            np.diag(self.S_r_t(lam_r, lam_t, lam_z) * O),
            self.S_r_z(lam_r, lam_t, lam_z) * O,

            np.diag(self.S_t_r(lam_r, lam_t, lam_z) * O),
            np.diag(self.S_t_t(lam_r, lam_t, lam_z)),
            self.S_t_z(lam_r, lam_t, lam_z) * O,

            self.S_z_r(lam_r, lam_t,  lam_z) * O,
            self.S_z_t(lam_r, lam_t,  lam_z) * O,
            self.S_z_z(lam_r, lam_t, lam_z)
        )





class Hyperelastic(Mechanics):
    """
    A class for a hyperelastic material.
    """
    def __init__(self):
        super().__init__()

    def build(self):
        """
        Builds the symbolic model by calculating the stresses
        and their derivatives
        """
        self.compute_stress()
        self.stress_derivatives()

    def compute_stress(self, homogeneous = False):
        """
        Method for computing the stresses from the elastic strain energy
        function W (defined in subclasses of the hyperelastic class)
        """

        if not(homogeneous):
            self.sig_r = 1 / self.J_0 * sp.diff(self.W, self.lam_r)
            self.sig_t = 1 / self.J_0 * sp.diff(self.W, self.lam_t)
            self.sig_z = 1 / self.J_0 * sp.diff(self.W, self.lam_z)
        else:
            self.sig_r = 1 / self.J_0 * sp.diff(self.W, self.lam_r)
            self.sig_t = self.sig_r
            self.sig_z = 1 / self.J_0 * sp.diff(self.W, self.lam_z)
