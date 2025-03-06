from .base_mechanics import Hyperelastic, np, sp

class FibreRecruitment(Hyperelastic):
    """
    A class for fibre-reinforced neo-Hookean materials.  The model
    accounts for fibre recruitment.  The averaging of the fibre
    orientation is carried out numerically using quadrature.
    """

    def __init__(self, pars = {}, distribution = 'linear', homogeneous = False):
        super().__init__()

        # Label
        self.label = 'Fibre recruitment'

        # Definition of constants in the model as SymPy symbols
        self.E_m = sp.Symbol('E_m')
        self.nu_m = sp.Symbol('nu_m')
        self.Phi_f = sp.Symbol('Phi_f')
        self.E_f = sp.Symbol('E_f')

        # Integration parameters
        self.k = sp.Symbol('k')
        self.N = 12
        self.Theta = 2 * sp.pi * self.k / self.N

        # Strain energy of the neo-Hookean matrix
        G = self.E_m / 2 / (1 + self.nu_m)
        lame = 2 * G * self.nu_m / (1 - 2 * self.nu_m)
        W_m = G / 2 * (self.I_1 - 3 - 2 * sp.log(self.J_t)) + lame / 2 * (self.J_t - 1)**2

        # Subtract off a small amount from lam_t to ensure lam_r > lam_t, 
        # which prevents singularities when lam_r \simeq lam_t
        lam_t = self.lam_t - 1e-4
        Lam_t = self.beta_r * lam_t

        # Compute the stretch and its mean over all angles
        if not(homogeneous):
            lam = sp.sqrt(self.Lam_r**2 * sp.cos(self.Theta)**2 + Lam_t**2 * sp.sin(self.Theta)**2)
            Lam = self.average(lam)
        else:
            lam = self.Lam_r
            Lam = lam

        # Compute the contribution to the energy from the fibres
        if distribution == 'triangle':
                
            lam_a = sp.Symbol('lam_a')
            lam_b = sp.Symbol('lam_b')
            lam_c = sp.Symbol('lam_c')

            w1 = ((2 * lam ** 2 + 4 * lam * lam_a) * sp.log(lam) + (-2 * lam ** 2 - 4 * lam * lam_a) * sp.log(lam_a) + (lam_a + 5 * lam) * (lam_a - lam)) / (-lam_c + lam_a) / (-lam_b + lam_a)
            w2 = (4 * (-lam_c + lam_a) * lam * (lam_b + lam / 2) * sp.log(lam) - 4 * lam * (lam_a + lam / 2) * (lam_b - lam_c) * sp.log(lam_a) - 4 * (-lam_b + lam_a) * lam * (lam_c + lam / 2) * sp.log(lam_c) + (-lam_c + lam_a) * ((lam_b - lam_c) * lam_a + lam_b * lam_c + 4 * lam * lam_b - 5 * lam ** 2)) / (-lam_b + lam_a) / (-lam_c + lam_a) / (lam_b - lam_c)
            w3 = (-4 * lam * (lam_a + lam / 2) * (lam_b - lam_c) * sp.log(lam_a) + 4 * (-lam_c + lam_a) * lam * (lam_b + lam / 2) * sp.log(lam_b) + (-lam_b + lam_a) * ((-2 * lam ** 2 - 4 * lam * lam_c) * sp.log(lam_c) + (lam_b - lam_c) * (-lam_c + lam_a))) / (-lam_b + lam_a) / (-lam_c + lam_a) / (lam_b - lam_c)

            if not(homogeneous):
                W1 = self.average(w1)
                W2 = self.average(w2)
                W3 = self.average(w3)
            else:
                W1 = w1
                W2 = w2
                W3 = w3

            W_f = self.E_f / 2 * sp.Piecewise(
                (0, Lam < lam_a),
                (W1, sp.And(lam_a < Lam, Lam < lam_c)),
                (W2, sp.And(lam_c < Lam, Lam < lam_b)),
                (W3, Lam > lam_b)
            )

        elif distribution == 'linear':
            lam_m = sp.Symbol('lam_m')

            w2 = (2*lam**2*lam_m - 2*lam**2*sp.log(lam) + 3*lam**2 - 4*lam*lam_m*sp.log(lam) - 4*lam - 2*lam_m + 1)/(lam_m**2 - 2*lam_m + 1)
            w3 = (2*lam**2*lam_m - 2*lam**2 - 2*lam*(lam + 2*lam_m)*sp.log(lam_m) - 4*lam - lam_m**2 + 2*lam_m*(2*lam + lam_m) - 2*lam_m + 1)/(lam_m**2 - 2*lam_m + 1)
            
            if not(homogeneous):
                W2 = self.average(w2)
                W3 = self.average(w3)
            else:
                W2 = w2
                W3 = w3

            W_f = self.E_f / 2 * sp.Piecewise(
                (0, Lam < 1),
                (W2, sp.And(1 < Lam, Lam < lam_m)),
                (W3, Lam > lam_m)
            )

        elif distribution == 'quartic':
            lam_m = sp.Symbol('lam_m')

            tmp = 3 * lam_m**2 + 4 * lam_m + 3
            f = self.E_f / 2 * ((lam-1)**4 * (5 * lam_m - 2 * lam - 3) / (lam_m-1)**3 / tmp)
            g = self.E_f / 2 * (10 * lam * (lam - lam_m - 1) / tmp + 1)
            
            if not(homogeneous):
                F = self.average(f)
                G = self.average(g)
            else:
                F = f; G = g

            # Final averaged strain energy of the fibres
            W_f = sp.Piecewise((0, Lam < 1), (F, sp.And(1 < Lam, Lam < lam_m)), (G, Lam > lam_m))

        else:
            raise Exception('ERROR: Unknown recruitment distribution')


        # Total strain energy
        self.W = (1 - self.Phi_f) * W_m + self.Phi_f * W_f

        # Build the model
        self.build()
    
    def average(self, f):
        """
        Computes the average over fibre angles using trapezoidal
        integration following Trefethen and Weideman, SIAM Review,
        Vol 56, No. 3, pp. 385-458, 2014
        """
        return 4 / self.N * (sp.summation(f, (self.k, 0, int(self.N/4))) - 
                            1/2 * f.subs(self.k, 0) - 1/2 * f.subs(self.k, int(self.N/4))
        )