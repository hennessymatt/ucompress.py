from .base_mechanics import Hyperelastic, np, sp

class NeoHookean(Hyperelastic):
    """
    A class for neo-Hookean material.  The strain-energy
    function is defined in the constructor along
    with any other parameters that are needed
    """

    def __init__(self):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.E_m = sp.Symbol('E_m')
        self.nu_m = sp.Symbol('nu_m')

        # Converting E and nu into G and the lame parameter
        G = self.E_m / 2 / (1 + self.nu_m)
        lame = 2 * G * self.nu_m / (1 - 2 * self.nu_m)

        # Hyperelastic strain energy
        self.W = G / 2 * (self.I_1 - 3 - 2 * sp.log(self.J_t)) + lame / 2 * (self.J_t - 1)**2

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
            np.zeros((N, N)),
            np.zeros(N),

            np.zeros((N, N)),
            np.diag(self.S_t_t(lam_r, lam_t, lam_z)),
            np.zeros(N),

            np.zeros(N),
            np.zeros(N),
            self.S_z_z(lam_r, lam_t, lam_z)
        )


