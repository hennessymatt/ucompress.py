from .base_mechanics import Hyperelastic, np, sp

class FibreReinforcedRecruitment(Hyperelastic):
    """
    Class for our new model
    """

    def __init__(self, pars = {}):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.G_m = sp.Symbol('G_m')
        self.alpha_f = sp.Symbol('alpha_f')
        self.E_f = sp.Symbol('E_f')
        self.lam_m = sp.Symbol('lam_m')

        # Integration parameters
        self.k = sp.Symbol('k')
        self.N = 12
        self.Theta = 2 * sp.pi * self.k / self.N

        # Strain energy of the neo-Hookean matrix
        W_nH = self.G_m / 2 * (self.I_1 - 2 * sp.log(self.J))

        # Subtract off a small amount from lam_t to ensure lam_r > lam_t, 
        # which prevents singularities when lam_r \simeq lam_t
        lam_t = self.lam_t - 1e-4

        # Compute the stretch and its mean
        lam = sp.sqrt(self.lam_r**2 * sp.cos(self.Theta)**2 + lam_t**2 * sp.sin(self.Theta)**2)
        lam_c = self.average(lam)

        # Compute the parts of the fibre strain energy
        tmp = 3 * self.lam_m**2 + 4 * self.lam_m + 3
        f = self.E_f / 2 * ((lam-1)**4 * (5 * self.lam_m - 2 * lam - 3) / (self.lam_m-1)**3 / tmp)
        g = self.E_f / 2 * (10 * lam * (lam - self.lam_m - 1) / tmp + 1)
        F = self.average(f)
        G = self.average(g)

        # Final averaged strain energy of the fibres
        W_f = sp.Piecewise((0, lam_c < 1), (F, sp.And(1 < lam_c, lam_c < self.lam_m)), (G, lam_c > self.lam_m))

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

