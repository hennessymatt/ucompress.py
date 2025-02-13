class PorosityCalculator():
    """
    Calculates the porosity of a material (with/without fibres)
    given the mass fraction of the hydrated state
    """

    def __init__(self, params):

        self.params = params

    def solve(self, have_fibres, update_params = False):

        params = self.params

        # solvent (water) mass fraction
        psi = params.physical["psi_0"]

        # solvent (water) density
        rho_w = params.physical["rho_w"]

        # matrix density
        rho_m = params.physical["rho_m"]

        # load fibre properties and calculate mean solid density
        if have_fibres:
            rho_f = params.physical["rho_f"]
            Phi_f = params.physical["Phi_f"]

            rho_s = Phi_f * rho_f + (1 - Phi_f) * rho_m

        # if no fibres, the solid density is the matrix density
        else:
            rho_s = rho_m

        # calculate relative density
        rho_r = rho_w / rho_s

        # now calculate the porosity
        phi = 1 / (1 + rho_r * (1 / psi - 1))

        # update the parameters
        if update_params:
            params.physical["phi_0"] = phi
            params.update()

            return params

        else:
            return phi