import matplotlib.pyplot as plt
import parameters

import ucompress as uc

import cProfile

pars = parameters.Parameters()

model = {
    "mechanics": uc.mechanics.NeoHookean(pars.physical),
    "permeability": uc.permeability.KozenyCarman(pars.physical),
}


problem = uc.solvers.DisplacementControlled(model, pars)
sol_instant = problem.initial_response()
print(sol_instant.F)
print(sol_instant.lam_z)
print(sol_instant.p)

problem.opts["monitor_convergence"] = False
problem.opts["newton_max_iterations"] = 10
sol = problem.solve() 

# # pr.disable()
# # pr.print_stats(sort = 'cumtime')




plt.figure()
plt.semilogx(sol.t[1:], sol.p[0,1:], '--')
# plt.semilogx(sol.t[1:], sol.u[-1, 1:])

# plt.figure()
# # plt.semilogx(pars.t[1:], old_problem.p[0,1:])
# # plt.semilogx(sol.t[1:], sol.p[0,1:], '--')
# # plt.semilogx(sol2.t[1:], sol2.p[0,1:], '-.')
# plt.semilogx(sol.t[1:], sol.lam_z[1:], '-.')


plt.show()