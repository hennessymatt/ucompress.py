import ucompress as uc
import numpy as np

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

        pars = uc.parameters.example_parameters.NeoHookean(nondim=nd)
        mech = uc.mechanics.NeoHookean()
        perm = uc.permeability.Constant()

        model = uc.base_models.Poroelastic(mech, perm, pars)

        exp = uc.experiments.ForceControlled(model, pars)

        sol = exp.transient_response(solver_opts)

        if nd:
            sol.redimensionalise(pars)

        t.append(sol.t)
        lam_z.append(sol.lam_z)


    assert np.linalg.norm(lam_z[0] - lam_z[1]) < 1e-8