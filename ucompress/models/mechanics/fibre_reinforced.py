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

    def __init__(self, homogeneous = False):
        super().__init__()

        # Label
        self.label = 'Fibre reinforced'

        # Definition of constants in the model as SymPy symbols
        self.E_m = sp.Symbol('E_m')
        self.nu_m = sp.Symbol('nu_m')
        self.Phi_f = sp.Symbol('Phi_f')
        self.E_f = sp.Symbol('E_f')

        # Converting E and nu into G and the lame parameter
        G = self.E_m / 2 / (1 + self.nu_m)
        lame = 2 * G * self.nu_m / (1 - 2 * self.nu_m)

        # Hyperelastic strain energy
        W_m = G / 2 * (self.I_1 - 3 - 2 * sp.log(self.J_t)) + lame / 2 * (self.J_t - 1)**2

        # Subtract off a small amount from lam_t to ensure lam_r > lam_t, 
        # which prevents singularities when lam_r \simeq lam_t
        lam_t = self.lam_t - 1e-4
        Lam_t = self.beta_r * lam_t

        # Strain energy of the fibres
        if not(homogeneous):
            W_f = self.E_f / 4 * (
                self.I_1_x - 8 * self.Lam_r / sp.pi * sp.elliptic_e(1 - Lam_t**2 / self.Lam_r**2) + 2
                )
        else:
            W_f = self.E_f / 2 * (self.Lam_r - 1)**2

        # Total strain energy
        self.W = (1 - self.Phi_f) * W_m + self.Phi_f * W_f

        # Update the SymPy -> SciPy conversion dictionary
        self.conversion_dict = {
            'elliptic_e': ellipe,
            'elliptic_k': ellipk
        }

        # Build the symbolic model
        self.build()
    