import ucompress as uc
import numpy as np

def test_nondim_displacement():
    """
    Tests that the non-dim of a standard neo-Hookean model
    with displacement-controlled loading works correctly
    """

    nondim = [False, True]

    t = []
    F = []

    solver_opts = {
        "jacobian": "analytical", # use analytical Jacobian for Newton iterations
        "monitor_convergence": False, # monitor convergence of newton iterations
        "newton_max_iterations": 10, # maximum number of newton iterations
        "abs_tol": 1e-8, # absolute error (of residual) convergence tolerance
        "rel_tol": 1e-8, # relative error (of residual) convergence tolerance
        "div_tol": 1e20
    }

    for n, nd in enumerate(nondim):

        pars = uc.parameters.example_parameters.NeoHookean(nondim=nd)
        mech = uc.mechanics.NeoHookean()
        perm = uc.permeability.Constant()

        model = uc.base_models.Poroelastic(mech, perm, pars)

        exp = uc.experiments.DisplacementControlled(model, pars)

        sol = exp.transient_response(solver_opts)

        if nd:
            sol.redimensionalise(pars)

        t.append(sol.t)
        F.append(sol.F)


    assert np.linalg.norm(t[0] - t[1]) < 1e-10
    assert np.linalg.norm(F[0] - F[1]) < 0.02
