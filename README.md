# UnconfinedCompression

Lightweight Python code for simulating the unconfined compression of 
nonlinear poroelastic materials.  The poroelastic sample
is assumed to remain cylindrical during compression.
The code uses Chebyshev spectral differentiation 
along with fully implicit time stepping.  An analytical Jacobian
is used in Newton iterations for speed.

Features of the code include:
* Displacement- and force-controlled loading
* Accounting for large deformation (finite strains)
* Neo-Hookean material responses
* Material reinforcement with a transversely isotropic fibre network
* Multiple models for the engagement of fibre network with deformation
* Deformation-dependent permeabilities
* Models for osmotic stresses and swelling (e.g. for hydrogels)
* Functions to fit stress-strain data

The code uses SymPy to generate exact derivatives of 
hyperelastic strain energies and permeabilities. This makes it
easy to extend and customise the material properties without
having to manually update the entries in the Jacobian matrix.