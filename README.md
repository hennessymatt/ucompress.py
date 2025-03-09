# ucompress.py

Lightweight Python code for simulating the unconfined compression of
cylindrical nonlinear poroelastic materials.  The poroelastic sample
is assumed to remain cylindrical during compression.

Features of the code include:
* Displacement- and force-controlled loading
* Finite strains and neo-Hookean material responses
* Material reinforcement with a transversely isotropic fibre network
* Models for the engagement of fibre network with deformation
* Deformation-dependent permeabilities
* Models for osmotic stresses and swelling (e.g. for hydrogels)
* Functions to fit stress-strain data


The code uses Chebyshev spectral differentiation 
along with fully implicit time stepping.  An analytical
Jacobian is automatically built using SymPy, allowing
for fast Newton iterations and easy generalisation
of the model. 

## Installing ucompress.py

ucompress.py is available on PyPi and can be easily installed using pip:

```
pip install ucompress
```

The dependencies are minimal, all you need is SciPy, NumPy, and SymPy.

## Getting started

The code below will simulate the force-controlled unconfined compression
of a neo-Hookean material with constant permeability.  The
axial stretch $\lambda_z$ is then plotted as a function of time
using matplotlib.

```python

import ucompress as uc
import matplotlib.pyplot as plt

pars = uc.parameters.example_parameters.NeoHookean()
mech = uc.mechanics.NeoHookean()
perm = uc.permeability.Constant()

model = uc.base_models.Poroelastic(mech, perm, pars)

problem = uc.experiments.ForceControlled(model, pars)
sol = problem.transient_response()

# plot the axial stretch vs time
plt.plot(sol.t, sol.lam_z)

```

## Learning more

To learn more about the capabilities of ucompress.py or to
how to use it, please work through the [tutorials](/tutorials/).