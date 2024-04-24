import matplotlib.pyplot as plt
import ucompress as uc
import numpy as np

pars = uc.parameters.example_parameters.NeoHookean()



model = {
    "mechanics": uc.mechanics.NeoHookean(pars),
    "permeability": uc.permeability.KozenyCarman(pars),
}


problem = uc.experiments.ForceControlled(model, pars)
sol_instant = problem.initial_response()
print(sol_instant.F)
print(sol_instant.lam_z)
print(sol_instant.p)

problem.opts["monitor_convergence"] = False
problem.opts["newton_max_iterations"] = 10
sol = problem.solve() 


plt.figure()
plt.semilogx(sol.t[1:], sol.p[0,1:], '--')

# D_fem = np.loadtxt('ucompress/tests/data/disp_data.csv', delimiter = ',')
D_fem = np.loadtxt('ucompress/tests/data/force_data.csv', delimiter = ',')
t_fem = D_fem[0]
p_fem = D_fem[1]

plt.semilogx(t_fem, p_fem, 'k')

plt.show()