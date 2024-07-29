# Mass Distribution Code for Strong Gravitational Lensing

## Description
The aim of this code is to reconstruct strong gravitational lenses using parammetric reconstruction with
lens plane optimization and the metropolis-hastings algorithm. The code is written in a recreative way 
meaning that, some functions are coded from scratch such as the Newton-Raphson method in order to, 
practice different numerical methods and techniques.

The current state of the code allows only to reconstruct sources for non-singular isothermal potentials
and point-like sources. The code is also being developped and currently its unit tests are being
written.

## Current problems with the code
- The Newton method for finding roots do not alwas converge and due to the asymptotic character of the lens
equation the method tends to go out of the domain and the code crashes.
- The convergence of the Metropolis-Hastings method has been correctly observed only for particular cases.
- The cost function is calculated by using the norm of the distances between the estimated and real 
positions while, the corect way to calculate the cost function is to use two cost functions, one for each
axis.
- The code only works with odd number of grid points for every axis.

## Things left to do
### Essencial to the code
- Finish the unit tests.
- Solve the problem of the newton methods by implementing boundary conditions.
- Assess the convergence of the Metropolis-Hastings method. Test other priors.
- Implement the correct Chi2.
- Resolve the parity issue of the number of grid points.

### To go further
- Implement extended sources.
- Implement the computation of the crtical lines, caustics and deformations of
the second order.
- Implement the calculation of the time delay, the intensity and the flux.
- Implement a rotation of the potential.
- Implement different potentials.
- Implement physical units.
- Implement the reconstruction method with other cost functions associated
with the flux and the time delays.
- Implement gaussian generation of images based on the fluxes and intensities.
- Make it a package.

## Contents
- main.py:
    Main script to be run.
- docs:
    Contains the documentation.
- output: 
    Contains the output files of the generated images.
- src:
    Contains all the modules necessary for the reconstruction.
    - calculus.py:
        Contains the purely mathematical functions e.g. gradient computation,
        Newton-Raphson root finding algorithm etc...
    - construct.py:
        Contains the source and the lens classes.
    - gen_params.py:
        Contains all the parameters needed for the generation of an image.
    - generate.py:
        Contains the code that generates an image.
    - rec_params.py:
        Loads the parameters for the reconstruction of the gravitational poten-
        tial.
    - reconstruct.py:
        Contains the algorithm that reconstruct the lens parameters.
    - statfunc.py:
        Contains the statistical functions and the functions needed for the 
        parametric reconstruction.
- tests:
    Contains the tests. The test files specify which files are being tested.

## How to use
Run the main.py file. Enter G if you only want to generate a new image,
enter GR if you want to generate the reconstruct the lens or, enter any
other key to abort. To generate without saving the file specify in the 
arguments of src.generate.Generate(save == False). To modify the lens 
parameters go to gen_params.py and modify them. The dispersion velocity 
and the ellipticity are both set to 1, 1 in order to run the tests. 
But they can be set to random, the code is written for the dispersion
velocity to be set between 0 and 5 and, the ellipticity between 0 
(excluded) and 1 (included).

