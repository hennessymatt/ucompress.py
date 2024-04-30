# UnconfinedCompression

Lightweight Python code for simulating the unconfined compression of 
nonlinear poroelastic materials.  The poroelastic sample
is assumed to remain cylindrical during compression.
The code uses Chebyshev spectral differentiation 
along with fully implicit time stepping.  The Jacobian matrix
for Newton iterations has been coded by hand for speed.

Features of the code include:
* Displacement- and force-controlled loading
* neo-Hookean and fibre-reinforced neo-Hookean material responses
* Constant and deformation-dependent permeabilities

The code uses SymPy to generate exact derivatives of 
hyperelastic strain energies and permeabilities. This makes it
easy to extend and customise the material properties without
having to manually update the entries in the Jacobian matrix.
