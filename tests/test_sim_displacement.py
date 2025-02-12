import ucompress as uc
import numpy as np
from parameters.NeoHookean import NeoHookean as Parameters

def test_sim_displacement():
    """
    Tests a displacement-controlled loading experiment
    by comparing to a FEniCS code
    """

    pars = Parameters(nondim = True)
    mech = uc.mechanics.NeoHookean()
    perm = uc.permeability.KozenyCarman()

    model = uc.base_models.Poroelastic(mech, perm, pars)

    exp = uc.experiments.DisplacementControlled(model, pars)
    sol = exp.transient_response()
    sol.redimensionalise(pars)

    data = np.loadtxt('tests/data/disp_data.csv', delimiter=',')
    p_0 = data[1]

    # check that the max pressure is the same
    assert abs(np.max(p_0) - np.max(sol.p[0,:])) < 1e-4
