from .base_mechanics import Hyperelastic, np, sp
from scipy.special import ellipk, ellipe

class FibreReinforcedInvariants(Hyperelastic):
    """
    Class for a fibre-reinforced neo-Hookean material.  The
    averaging over the fibre angles is done exactly, resulting
    in appearance of complete elliptic integrals in the
    strain energy.  The strain energy is formulated in terms
    of the invariants.
    """
    
    def __init__(self, pars = {}, formulation = "stretches"):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.G_m = sp.Symbol('G_m')
        self.alpha_f = sp.Symbol('alpha_f')
        self.G_f = sp.Symbol('G_f')

        # In-plane invariants
        I_1_x = self.lam_r**2 + self.lam_t**2
        I_2_x = self.lam_r**2 * self.lam_t**2

        # Strain energy of the neo-Hookean matrix
        W_nH = self.G_m / 2 * (self.I_1 - 2 * sp.log(self.J))

        # Strain energy of the fibres
        tmp1 = sp.sqrt(I_1_x**2 - 4 * I_2_x + 1e-8)
        tmp2 = I_1_x + tmp1
        W_f = self.G_f / 4 * (
            I_1_x + 8 * sp.sqrt(2) / sp.pi / sp.sqrt(tmp2) * sp.elliptic_k(2 * tmp1 / tmp2) - 6
            )

        # Total strain energy
        self.W = (1 - self.alpha_f) * W_nH + self.alpha_f * W_f

        # Update conversion dictionary (SymPy to SciPy)
        self.conversion_dict = {'elliptic_k': ellipk, 'elliptic_e': ellipe}

        # compute stresses, stress derivatives, and convert to NumPy expressions
        self.compute_stress()
        self.stress_derivatives()
        self.lambdify(pars)

    
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

