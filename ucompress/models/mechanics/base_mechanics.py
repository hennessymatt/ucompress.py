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
        self.lam_r = sp.Symbol('lam_r')
        self.lam_t = sp.Symbol('lam_t')
        self.lam_z = sp.Symbol('lam_z')

        self.J = self.lam_r * self.lam_t * self.lam_z
        self.I_1 = self.lam_r**2 + self.lam_t**2 + self.lam_z**2


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
          
    def lambdify(self, pars, conversion_dict ={}):
        """
        Turns the SymPy expressions for the stresses and
        their derivatives into fast NumPy functions

        conversion_dict is a dictionary that contains information
        about how to convert any special SymPy functions (e.g elliptic
        integrals) into SciPy functions
        """

        # Define the arguments of the NumPy function
        args = [self.lam_r, self.lam_t, self.lam_z]

        # Type of function to create
        translation = [conversion_dict, 'numpy']

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
        Numerically evaluates the stress derivatives and
        returns them
        """
        return (
            np.diag(self.S_r_r(lam_r, lam_t, lam_z)),
            np.diag(self.S_r_t(lam_r, lam_t, lam_z)),
            self.S_r_z(lam_r, lam_t, lam_z),

            np.diag(self.S_t_r(lam_r, lam_t, lam_z)),
            np.diag(self.S_t_t(lam_r, lam_t, lam_z)),
            self.S_t_z(lam_r, lam_t, lam_z),

            self.S_z_r(lam_r, lam_t, lam_z),
            self.S_z_t(lam_r, lam_t, lam_z),
            self.S_z_z(lam_r, lam_t, lam_z)
        )



class Hyperelastic(Mechanics):
    """
    A class for a hyperelastic material.
    """
    def __init__(self):
        super().__init__()

    def compute_stress(self):
        """
        Method for computing the stresses from the elastic strain energy
        function W (defined in subclasses of the hyperelastic class)
        """
        self.sig_r = sp.diff(self.W, self.lam_r)
        self.sig_t = sp.diff(self.W, self.lam_t)
        self.sig_z = sp.diff(self.W, self.lam_z)