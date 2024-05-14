from .base_mechanics import Hyperelastic, np, sp

class FibreReinforcedQuad(Hyperelastic):
    """
    Class for a fibre-reinforced neo-Hookean material.
    This approach uses trapezoidal quadrature to numerically
    approximate the average over the fibre angles.
    """

    def __init__(self, pars = {}):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.G_m = sp.Symbol('G_m')
        self.alpha_f = sp.Symbol('alpha_f')
        self.G_f = sp.Symbol('G_f')

        # Integration parameters
        self.k = sp.Symbol('k')
        self.N = 12
        self.Theta = 2 * sp.pi * self.k / self.N

        # Strain energy of the neo-Hookean matrix
        W_nH = self.G_m / 2 * (self.I_1 - 2 * sp.log(self.J))

        # Subtract off a small amount from lam_t to ensure lam_r > lam_t, 
        # which prevents singularities when lam_r \simeq lam_t
        lam_t = self.lam_t - 1e-4

        # Define the stretch
        lam = sp.sqrt(self.lam_r**2 * sp.cos(self.Theta)**2 + lam_t**2 * sp.sin(self.Theta)**2)
        
        # Define the elastic energy of the fibres and its average
        w_f = self.G_f / 2 * (lam**2 + 2 / lam - 3)
        W_f = self.average(w_f)

        # Total strain energy
        self.W = (1 - self.alpha_f) * W_nH + self.alpha_f * W_f

        # compute stresses, stress derivatives, and convert to NumPy expressions
        self.compute_stress()
        self.stress_derivatives()
        self.lambdify(pars)

    def average(self, f):
        """
        Computes the average over fibre angles using trapezoidal
        integration following Trefethen and Weideman, SIAM Review,
        Vol 56, No. 3, pp. 385–458, 2014
        """
        return 4 / self.N * (sp.summation(f, (self.k, 0, int(self.N/4))) - 
                            1/2 * f.subs(self.k, 0) - 1/2 * f.subs(self.k, int(self.N/4))
        )

    
    def eval_stress_derivatives(self, lam_r, lam_t, lam_z):
        """
        Overloads the method for evaluating the stress derivatives
        to zero out certain entries and ensure the outputs have
        the correct shape.
        """

        N = len(lam_r)

        return (
            np.diag(self.S_r_r(lam_r, lam_t, lam_z)),
            np.diag(self.S_r_t(lam_r, lam_t, lam_z)),
            np.zeros(N),

            np.diag(self.S_t_r(lam_r, lam_t, lam_z)),
            np.diag(self.S_t_t(lam_r, lam_t, lam_z)),
            np.zeros(N),

            np.zeros(N),
            np.zeros(N),
            self.S_z_z(lam_r, lam_t, lam_z)
        )

