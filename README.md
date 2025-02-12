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