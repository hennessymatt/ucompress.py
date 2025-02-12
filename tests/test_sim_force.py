import ucompress as uc
import numpy as np
from parameters.NeoHookean import NeoHookean as Parameters

def test_sim_force():
    """
    Tests a force-controlled loading experiment
    by comparing to a FEniCS code
    """
    pars = Parameters(nondim=True)
    pars.update("F", -2)
    mech = uc.mechanics.NeoHookean()
    perm = uc.permeability.KozenyCarman()

    model = uc.base_models.Poroelastic(mech, perm, pars)

    exp = uc.experiments.ForceControlled(model, pars)
    sol = exp.transient_response()
    sol.redimensionalise(pars)

    data = np.loadtxt('tests/data/force_data.csv', delimiter=',')
    t = data[0]
    p_0 = data[1]
    lam_z = data[2]

    # check the times are the same
    assert np.linalg.norm(t - sol.t) < 1e-8

    # check the axial stretch is the same
    assert np.linalg.norm(lam_z[1:] - sol.lam_z[1:]) < 1e-3

