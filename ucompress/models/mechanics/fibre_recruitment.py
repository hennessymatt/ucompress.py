from .base_mechanics import Hyperelastic, np, sp

class FibreRecruitment(Hyperelastic):
    """
    A class for fibre-reinforced neo-Hookean materials.  The model
    accounts for fibre recruitment.  The averaging of the fibre
    orientation is carried out numerically using quadrature.
    """

    def __init__(self, pars = {}, distribution = 'triangle'):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.E_m = sp.Symbol('E_m')
        self.nu_m = sp.Symbol('nu_m')
        self.alpha_f = sp.Symbol('alpha_f')
        self.E_f = sp.Symbol('E_f')

        # Integration parameters
        self.k = sp.Symbol('k')
        self.N = 12
        self.Theta = 2 * sp.pi * self.k / self.N

        # Strain energy of the neo-Hookean matrix
        G_m = self.E_m / 2 / (1 + self.nu_m)
        l_m = 2 * G_m * self.nu_m / (1 - 2 * self.nu_m)
        W_m = G_m / 2 * (self.I_1 - 3 - 2 * sp.log(self.J)) + l_m / 2 * (self.J - 1)**2

        # Subtract off a small amount from lam_t to ensure lam_r > lam_t, 
        # which prevents singularities when lam_r \simeq lam_t
        lam_t = self.lam_t - 1e-4

        # Compute the stretch and its mean
        lam = sp.sqrt(self.lam_r**2 * sp.cos(self.Theta)**2 + lam_t**2 * sp.sin(self.Theta)**2)
        Lam = self.average(lam)

        # Compute the parts of the fibre strain energy

        if distribution == 'triangle':
                
            lam_a = sp.Symbol('lam_a')
            lam_b = sp.Symbol('lam_b')
            lam_c = sp.Symbol('lam_c')

            A = sp.Piecewise(
                (0, Lam < lam_a),
                (-lam_a**2 / (lam_b - lam_a) / (lam_c - lam_a), sp.And(lam_a < Lam, Lam < lam_c)),
                (lam_c**2 / (lam_c - lam_a) / (lam_b - lam_c) - lam_a**2  / (lam_b - lam_a) / (lam_c - lam_a), sp.And(lam_c < Lam, Lam < lam_b)),
                (-1, Lam > lam_b)
            )

            B = sp.Piecewise(
                (0, Lam < lam_a),
                (2 * lam_a * sp.log(lam_a) / (lam_b - lam_a) / (lam_c - lam_a), sp.And(lam_a < Lam, Lam < lam_c)),
                (2 * lam_a * sp.log(lam_a) / (lam_b - lam_a) / (lam_c - lam_a) - 2 * lam_c * sp.log(lam_c) / (lam_c - lam_a) / (lam_b - lam_c), sp.And(lam_c < Lam, Lam < lam_b)),
                (2 * lam_a * sp.log(lam_a) / (lam_b - lam_a) / (lam_c - lam_a) - 2 * lam_c * sp.log(lam_c) / (lam_c - lam_a) / (lam_b - lam_c) + 2 * lam_b * sp.log(lam_b) / (lam_b - lam_a) / (lam_b - lam_c), Lam > lam_b)
            )

            C = sp.Piecewise(
                (0, Lam < lam_a),
                (1 / (lam_b - lam_a) / (lam_c - lam_a), sp.And(lam_a < Lam, Lam < lam_c)),
                (-1 / (lam_b - lam_a) / (lam_b - lam_c), sp.And(lam_c < Lam, Lam < lam_b)),
                (0, Lam > lam_b)
            )

            D = sp.Piecewise(
                (0, Lam < lam_a),
                (-2 * lam_a / (lam_b - lam_a) / (lam_c - lam_a), sp.And(lam_a < Lam, Lam < lam_c)),
                (2 * lam_b / (lam_b - lam_a) / (lam_b - lam_c), sp.And(lam_c < Lam, Lam < lam_b)),
                (0, Lam > lam_b)
            )

            W_f = self.E_f * (A * sp.log(Lam) + (B - D) * Lam + C / 2 * Lam**2 + D * Lam * sp.log(Lam))

        elif distribution == 'quartic':
            lam_m = sp.Symbol('lam_m')

            tmp = 3 * lam_m**2 + 4 * lam_m + 3
            f = self.E_f / 2 * ((lam-1)**4 * (5 * lam_m - 2 * lam - 3) / (lam_m-1)**3 / tmp)
            g = self.E_f / 2 * (10 * lam * (lam - lam_m - 1) / tmp + 1)
            F = self.average(f)
            G = self.average(g)

            # Final averaged strain energy of the fibres
            W_f = sp.Piecewise((0, Lam < 1), (F, sp.And(1 < Lam, Lam < lam_m)), (G, Lam > lam_m))


        # Total strain energy
        self.W = (1 - self.alpha_f) * W_m + self.alpha_f * W_f

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

