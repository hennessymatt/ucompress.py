import matplotlib.pyplot as plt
import parameters

import mechanics
import permeability
import solver

import cProfile

pars = parameters.StandardParameters()



model = {
    "mechanics": mechanics.FibreReinforcedNH({"G_m": 1, "G_f": 100, "alpha": 0.5}),
    "permeability": permeability.KozenyCarman({"k_0": 1, "phi_0": pars.phi_0}),
}


problem = solver.ForceControlled(model, pars)
sol_instant = problem.initial_response()
print(sol_instant.lam_z)
print(sol_instant.p)
sol = problem.solve({"monitor_convergence": False})

# pr.disable()
# pr.print_stats(sort = 'cumtime')




plt.figure()
plt.semilogx(pars.t[1:], sol.p[0,1:], '--')

plt.figure()
# plt.semilogx(pars.t[1:], old_problem.p[0,1:])
# plt.semilogx(sol.t[1:], sol.p[0,1:], '--')
# plt.semilogx(sol2.t[1:], sol2.p[0,1:], '-.')
plt.semilogx(sol.t[1:], sol.lam_z[1:], '-.')


plt.show()