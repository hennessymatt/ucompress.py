from .base_mechanics import Hyperelastic, np, sp

class NeoHookean(Hyperelastic):
    """
    A class for neo-Hookean material.  The strain-energy
    function is defined in the constructor along
    with any other parameters that are needed
    """

    def __init__(self, pars = {}):
        super().__init__()

        # Definition of constants in the model as SymPy symbols
        self.G_m = sp.Symbol('G_m')

        # Hyperelastic strain energy
        self.W = self.G_m/2 * (self.I_1 - 2 * sp.log(self.J))

        # Use SymPy to compute the stresses and their derivatives
        self.compute_stress()
        self.stress_derivatives()

        # Convert the SymPy expressions into NumPy arrays
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
            np.zeros((N, N)),
            np.zeros(N),

            np.zeros((N, N)),
            np.diag(self.S_t_t(lam_r, lam_t, lam_z)),
            np.zeros(N),

            np.zeros(N),
            np.zeros(N),
            self.S_z_z(lam_r, lam_t, lam_z)
        )


