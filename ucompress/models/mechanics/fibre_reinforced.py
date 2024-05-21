from .base_mechanics import Hyperelastic, np, sp
from scipy.special import ellipk, ellipe

class FibreReinforced(Hyperelastic):
    """
    Class for a fibre-reinforced neo-Hookean material.  The
    averaging over the fibre angles is done exactly, resulting
    in appearance of complete elliptic integrals in the
    strain energy.  The strain energy is formulated in terms
    of the stretches.
    """

    def __init__(self):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.E_m = sp.Symbol('E_m')
        self.nu_m = sp.Symbol('nu_m')
        self.alpha_f = sp.Symbol('alpha_f')
        self.E_f = sp.Symbol('E_f')

        # Converting E_m and nu_m into mu_m and l_m
        G_m = self.E_m / 2 / (1 + self.nu_m)
        l_m = 2 * G_m * self.nu_m / (1 - 2 * self.nu_m)

        # Hyperelastic strain energy of the matrix
        W_m = G_m / 2 * (self.I_1 - 3 - 2 * sp.log(self.J)) + l_m / 2 * (self.J - 1)**2

        # In-plane invariants
        I_1_x = self.lam_r**2 + self.lam_t**2

        # Subtract off a small amount from lam_t to ensure lam_r > lam_t, 
        # which prevents singularities when lam_r \simeq lam_t
        lam_t = self.lam_t - 1e-4

        # Strain energy of the fibres
        W_f = self.E_f / 4 * (
            I_1_x - 8 * self.lam_r / sp.pi * sp.elliptic_e(1 - lam_t**2 / self.lam_r**2) + 2
            )

        # Total strain energy
        self.W = (1 - self.alpha_f) * W_m + self.alpha_f * W_f

        # Update the conversion dictionary
        self.conversion_dict = {
            'elliptic_e': ellipe,
            'elliptic_k': ellipk
        }

        # Build the symbolic model
        self.build()
    
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

