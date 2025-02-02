"""

"""

from SPI2py.models.geometry.spherical_decomposition import pseudo_mdbd

# Perform MDBD on each part
num_spheres = 1000
min_radius = 0.01
meshgrid_increment = 75


pseudo_mdbd('models/',
            'stl/coolant_particle_filter.stl',
            'mdbd/coolant_particle_filter.xyzr',
            n_spheres=num_spheres,
            r_min=min_radius,
            n_steps=meshgrid_increment,
            plot=True)

pseudo_mdbd('models/',
            'stl/degas_bottle.stl',
            'mdbd/degas_bottle.xyzr',
            n_spheres=num_spheres,
            r_min=min_radius,
            n_steps=meshgrid_increment,
            plot=True)


pseudo_mdbd('models/',
            'stl/heater_core.stl',
            'mdbd/heater_core.xyzr',
            n_spheres=num_spheres,
            r_min=min_radius,
            n_steps=meshgrid_increment,
            plot=True)

pseudo_mdbd('models/',
            'stl/ion_exchanger.stl',
            'mdbd/ion_exchanger.xyzr',
            n_spheres=num_spheres,
            r_min=min_radius,
            n_steps=meshgrid_increment,
            plot=True)

pseudo_mdbd('models/',
            'stl/radiator.stl',
            'mdbd/radiator.xyzr',
            n_spheres=num_spheres,
            r_min=min_radius,
            n_steps=meshgrid_increment,
            plot=True)

pseudo_mdbd('models/',
            'stl/weg_heater.stl',
            'mdbd/weg_heater.xyzr',
            n_spheres=num_spheres,
            r_min=min_radius,
            n_steps=meshgrid_increment,
            plot=True)
