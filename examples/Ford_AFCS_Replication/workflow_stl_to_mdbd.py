"""

"""

import pyvista as pv
from SPI2py.models.mechanics.maximal_disjoint_ball_decomposition import mdbd

# Create an STL file for each part
radiator_and_ion_exchanger = pv.Cube(bounds=(0.0, 2.850, 0.0, 0.830, 0.0, 0.830))
radiator_and_ion_exchanger.save('models/stl/radiator_and_ion_exchanger.stl')

pump = pv.Cube(bounds=(0.0, 0.450, 0.0, 0.450, 0.0, 0.450))
pump.save('models/stl/pump.stl')

particle_filter = pv.Cube(bounds=(0.0, 0.489, 0.0, 0.330, 0.0, 0.330))
particle_filter.save('models/stl/particle_filter.stl')

fuel_cell_stack = pv.Cube(bounds=(0.0, 1.800, 0.0, 1.600, 0.0, 1.600))
fuel_cell_stack.save('models/stl/fuel_cell_stack.stl')

WEG_heater_and_pump = pv.Cube(bounds=(0.0, 1.655, 0.0, 1.833, 0.0, 1.833))
WEG_heater_and_pump.save('models/stl/WEG_heater_and_pump.stl')

heater_core = pv.Cube(bounds=(0.0, 1.345, 0.0, 0.460, 0.0, 0.460))
heater_core.save('models/stl/heater_core.stl')

# Perform MDBD on each part
num_spheres = 1000
min_radius = 0.001
meshgrid_increment = 100

mdbd('models/',
     'stl/radiator_and_ion_exchanger.stl',
     'mdbd/radiator_and_ion_exchanger.xyzr',
     num_spheres=num_spheres,
     min_radius=min_radius,
     meshgrid_increment=meshgrid_increment,
     plot=False)

mdbd('models/',
        'stl/pump.stl',
        'mdbd/pump.xyzr',
        num_spheres=num_spheres,
        min_radius=min_radius,
        meshgrid_increment=meshgrid_increment,
        plot=False)

mdbd('models/',
        'stl/particle_filter.stl',
        'mdbd/particle_filter.xyzr',
        num_spheres=num_spheres,
        min_radius=min_radius,
        meshgrid_increment=meshgrid_increment,
        plot=False)

mdbd('models/',
        'stl/fuel_cell_stack.stl',
        'mdbd/fuel_cell_stack.xyzr',
        num_spheres=num_spheres,
        min_radius=min_radius,
        meshgrid_increment=meshgrid_increment,
        plot=False)

mdbd('models/',
        'stl/WEG_heater_and_pump.stl',
        'mdbd/WEG_heater_and_pump.xyzr',
        num_spheres=num_spheres,
        min_radius=min_radius,
        meshgrid_increment=meshgrid_increment,
        plot=False)

mdbd('models/',
        'stl/heater_core.stl',
        'mdbd/heater_core.xyzr',
        num_spheres=num_spheres,
        min_radius=min_radius,
        meshgrid_increment=meshgrid_increment,
        plot=False)