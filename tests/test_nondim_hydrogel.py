import ucompress as uc
import numpy as np
from parameters.hydrogel import Hydrogel

def test_nondim_force():
    """
    Tests that the non-dim of a standard neo-Hookean model
    with displacement-controlled loading works correctly
    """

    nondim = [False, True]

    solver_opts = {
        "jacobian": "analytical", # use analytical Jacobian for Newton iterations
        "monitor_convergence": False, # monitor convergence of newton iterations
        "newton_max_iterations": 10, # maximum number of newton iterations
        "abs_tol": 1e-8, # newton convergence tolerance
        "rel_tol": 1e-8,
        "div_tol": 1e20
    }

    t = []
    lam_z = []

    for nd in nondim:

        if not(nd):
            solver_opts["abs_tol"] = 1e-2

        pars = Hydrogel(nondim=nd)
        mech = uc.mechanics.NeoHookean()
        perm = uc.permeability.Constant()
        os = uc.osmosis.FloryHuggins()

        model = uc.base_models.Hydrogel(mech, perm, os, pars)

        # hydration
        exp = uc.experiments.Hydration(model, pars)
        beta_r, beta_z, phi_0 = exp.steady_response()

        pars.update("beta_r", beta_r)
        pars.update("beta_z", beta_z)
        pars.update('phi_0', phi_0)
        pars.update('lam_z', 1)
        model.assign(pars)

        # force-controlled exp
        exp = uc.experiments.ForceControlled(model, pars)

        sol = exp.transient_response(solver_opts)

        if nd:
            sol.redimensionalise(pars)

        t.append(sol.t)
        lam_z.append(sol.lam_z)


    assert np.linalg.norm(lam_z[0] - lam_z[1]) < 1e-2