import ucompress as uc
import numpy as np

def test_jacobian():
    """
    Compares the analytical Jacobian to a numerical Jacobian computed
    using finite differences
    """

    pars = uc.parameters.example_parameters.Hydrogel(nondim = True)
    pars.update('beta_r', 2)
    pars.update('beta_z', 2)
    pars.update('phi_0', 0.7)
    
    mech = uc.mechanics.NeoHookean()
    os = uc.osmosis.FloryHuggins()
    perm = uc.permeability.KozenyCarman()

    model = uc.base_models.Hydrogel(mech, perm, os, pars)

    exp = uc.experiments.ForceControlled(model, pars)

    # Set the current and previous displacement
    exp.u = exp.r**2
    exp.u_old = 0.9 * exp.r**2

    # Compute/set the stretches
    exp.lam_r, exp.lam_t =  exp.compute_stretches(exp.u)
    exp.lam_r_old, exp.lam_t_old = exp.compute_stretches(exp.u_old)
    exp.lam_z = pars.physical["lam_z"]
    exp.lam_z_old = 0.9 * exp.lam_z

    # Set the pressure
    exp.p = np.exp(exp.r)

    # Set thet ime step
    exp.dt = 1e-2

    # Compute J = det(F) and the residual
    exp.compute_J()
    exp.build_residual()

    # compute the numerical Jacobian using finite differences
    exp.numerical_jacobian()
    J_n = exp.JAC.copy()

    # compute the analytical Jacobian
    exp.analytical_jacobian()
    J_a = exp.JAC.copy()

    # compute the error row-by-row and print the result
    for i in range(2*exp.N + 1):
        diff = np.linalg.norm(J_a[i,:] - J_n[i,:]) / np.linalg.norm(J_n[i,:], ord = np.inf)
        assert diff < 1e-2
