# What is FLEXPART?

FLEXPART (FLEXible PARTicle dispersion model) is a widely used Lagrangian Particle Dispersion Model (LPDM) designed for simulating the atmospheric transport, diffusion, and deposition of tracers. It operates by tracking the trajectories of a large number of 'particles' released into the atmosphere based on 3D wind fields from an Eulerian model. It also includes parameterizations for convection and turbulence. FLEXPART can be used for various applications (e.g., pollutant releases, wildfire smoke, greenhouse gases, or moisture and heat transport).

More information on FLEXPART, including example applications, can be found on [flexpart.eu](https://www.flexpart.eu/).

Technical details on how to install and run FLEXPART can be found in the [documentation](https://flexpart.img.univie.ac.at/docs/).

# What is FLEXWEB?

FLEXWEB is a web service for running FLEXPART online. The goal of FLEXWEB is to make FLEXPART more accessible to users who do not have the necessary infrastructure or technical expertise to install and run the model.

FLEXWEB is accessible at [https://flexweb.wolke.img.univie.ac.at/](https://flexweb.wolke.img.univie.ac.at/).

If automatic plotting is enabled, FLEXWEB plots the results and displays them. There are three plot types:
- Plume: The concentration of particles at each time step
- Sum: The time-integrated values of the plume
- Contour: The 50th percentile of the particle concentration at different time steps.
