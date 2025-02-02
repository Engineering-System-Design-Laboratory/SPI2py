import numpy as np
from jax import jacfwd

import openmdao.api as om
from openmdao.api import ExplicitComponent, Group

from ..models.utilities.inputs import read_xyzr_file
from ..models.utilities.aggregation import kreisselmeier_steinhauser_max
from ..models.computational_geometry.rigid_body_transformations import assemble_transformation_matrix, \
    apply_transformation_matrix
from .projection import Mesh, calculate_pseudo_densities
from ..models.projection.project_interconnects_vectorized import calculate_combined_densities

class SpatialConfiguration(Group):

    def initialize(self):
        self.options.declare('input_dict', types=dict)

    def setup(self):
        input_dict = self.options['input_dict']

        components_dict = input_dict['components']
        interconnects_dict = input_dict['interconnects']

        # Check if the components dictionary is empty
        if len(components_dict) > 0:

            components = om.Group()

            # Create the components
            components_dict = input_dict['components']
            for i, key in enumerate(components_dict.keys()):
                description = components_dict[key]['description']
                filepath = components_dict[key]['filepath']
                n_spheres = components_dict[key]['n_spheres']
                ports = components_dict[key]['ports']
                color = components_dict[key]['color']

                sphere_positions, sphere_radii = read_xyzr_file(filepath, num_spheres=n_spheres)

                component = Component(description=description,
                                      color=color,
                                      sphere_positions=sphere_positions,
                                      sphere_radii=sphere_radii,
                                      ports=ports)

                components.add_subsystem(f'comp_{i}', component)

            self.add_subsystem('components', components)


        # Check if the interconnects dictionary is empty
        if len(interconnects_dict) > 0:

            interconnects = om.Group()

            for i, key in enumerate(interconnects_dict.keys()):

                component_1 = interconnects_dict[key]['component_1']
                port_1 = interconnects_dict[key]['port_1']
                component_2 = interconnects_dict[key]['component_2']
                port_2 = interconnects_dict[key]['port_2']
                n_segments = interconnects_dict[key]['n_segments']
                radius = interconnects_dict[key]['radius']
                color = interconnects_dict[key]['color']

                interconnect = Interconnect(n_segments=n_segments,
                                            radius=radius,
                                            color=color)

                interconnects.add_subsystem(f'int_{i}', interconnect)

                # Connect the interconnects to the components
                self.connect(f'components.comp_{component_1}.transformed_ports',
                             f'interconnects.int_{i}.start_point',
                             src_indices=om.slicer[port_1, :])
                self.connect(f'components.comp_{component_2}.transformed_ports',
                             f'interconnects.int_{i}.end_point',
                             src_indices=om.slicer[port_2, :])

            self.add_subsystem('interconnects', interconnects)

        # Create the system
        system = System(n_projections=len(components_dict) + len(interconnects_dict), rho_min=1e-3)
        self.add_subsystem('system', system)


        # Connect the components to the system
        i = 0
        for j in range(len(components_dict)):
            self.connect(f'components.comp_{j}.pseudo_densities',
                          f'system.pseudo_densities_{i}')
            i += 1

        # Connect the interconnects to the system
        for j in range(len(interconnects_dict)):
            self.connect(f'interconnects.int_{j}.pseudo_densities',
                          f'system.pseudo_densities_{i}')
            i += 1


class Component(ExplicitComponent):

    def initialize(self):
        self.options.declare('description', types=str)
        self.options.declare('color', types=str)
        self.options.declare('sphere_positions', types=list)
        self.options.declare('sphere_radii', types=list)
        self.options.declare('ports', types=list)

    def setup(self):

        # Unpack the options
        sphere_positions = self.options['sphere_positions']
        sphere_radii = self.options['sphere_radii']
        ports = self.options['ports']

        # Convert the lists to numpy arrays
        sphere_positions = np.array(sphere_positions).reshape(-1, 3)
        sphere_radii = np.array(sphere_radii).reshape(-1, 1)
        ports = np.array(ports).reshape(-1, 3)

        # Define the input shapes
        self.add_input('sphere_positions', val=sphere_positions)
        self.add_input('sphere_radii', val=sphere_radii)
        self.add_input('ports', val=ports)
        self.add_input('translation', val=np.array([[0.0, 0.0, 0.0]]))
        self.add_input('rotation', val=np.array([[0.0, 0.0, 0.0]]))

        # Mesh Inputs
        self.add_input('element_length', val=0)
        self.add_input('centers', shape_by_conn=True)
        self.add_input('element_bounds', shape_by_conn=True)
        self.add_input('element_sphere_positions', shape_by_conn=True)
        self.add_input('element_sphere_radii', shape_by_conn=True)

        # Outputs:
        self.add_output('transformed_sphere_positions', val=sphere_positions)
        self.add_output('transformed_sphere_radii', val=sphere_radii)
        self.add_output('transformed_ports', val=ports)

        # Outputs: Projections
        # Define the outputs

        self.add_output('pseudo_densities',
                        compute_shape=lambda shapes: (shapes['centers'][0], shapes['centers'][1], shapes['centers'][2]))

        # Define diagnostic outputs
        self.add_output('mesh_kernel_volume_error', val=0.0, desc="How accurately the mesh kernel represents the element volume")
        self.add_output('projection_volume_error', val=0.0, desc='How accurately the projection represents the object')



    # def setup_partials(self):
    #     self.declare_partials('transformed_ports', ['translation', 'rotation'])
    #     self.declare_partials('pseudo_densities', ['translation', 'rotation'])

    def compute(self, inputs, outputs):

        # Get the input variables
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        port_positions = inputs['ports']
        translation = inputs['translation']
        rotation = inputs['rotation']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        element_bounds = inputs['element_bounds']
        sample_points = inputs['element_sphere_positions']
        sample_radii = inputs['element_sphere_radii']
        # volume = jnp.array(inputs['volume'])

        # Calculate the transformed sphere positions and port positions
        sphere_positions_transformed, ports_transformed = self._compute_primal(sphere_positions, port_positions,translation, rotation)

        # Compute the pseudo-densities
        pseudo_densities = self._project(sphere_positions_transformed, sphere_radii, sample_points, sample_radii, element_bounds)

        # # Define diagnostic outputs
        # self.add_output('mdbd_volume', val=np.sum((4 / 3) * np.pi * sphere_radii ** 3))
        # self.add_output('projected_volume', val=0.0)

        # Set the outputs
        outputs['transformed_sphere_positions'] = sphere_positions_transformed
        outputs['transformed_sphere_radii'] = sphere_radii
        outputs['transformed_ports'] = ports_transformed
        outputs['pseudo_densities'] = pseudo_densities

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the input variables
    #     sphere_positions = inputs['sphere_positions']
    #     sphere_radii = inputs['sphere_radii']
    #     ports = inputs['ports']
    #     translation = inputs['translation']
    #     rotation = inputs['rotation']
    #
    #     # Convert the input variables to Jax arrays
    #     sphere_positions = jnp.array(sphere_positions)
    #     sphere_radii = jnp.array(sphere_radii)
    #     ports = jnp.array(ports)
    #     translation = jnp.array(translation)
    #     rotation = jnp.array(rotation)
    #
    #     # Define the Jacobian matrices using PyTorch Autograd
    #     jac_sphere_positions = jacfwd(self.compute_transformation, argnums=(1, 2))
    #     jac_ports = jacfwd(self.compute_transformation, argnums=(1, 2))
    #
    #     # Evaluate the Jacobian matrices
    #     jac_sphere_positions_val = jac_sphere_positions(sphere_positions, translation, rotation)
    #     jac_ports_val = jac_ports(ports, translation, rotation)
    #
    #     # Slice the Jacobian matrices
    #     grad_sphere_positions_translation = jac_sphere_positions_val[0]
    #     grad_sphere_positions_rotation = jac_sphere_positions_val[1]
    #     grad_ports_translation = jac_ports_val[0]
    #     grad_ports_rotation = jac_ports_val[1]
    #
    #     # Set the outputs
    #     partials['transformed_sphere_positions', 'translation'] = grad_sphere_positions_translation
    #     partials['transformed_sphere_positions', 'rotation'] = grad_sphere_positions_rotation
    #     partials['transformed_ports', 'translation'] = grad_ports_translation
    #     partials['transformed_ports', 'rotation'] = grad_ports_rotation

    @staticmethod
    def _compute_primal(sphere_positions, port_positions, translation, rotation):

        # Assemble the transformation matrix
        t = assemble_transformation_matrix(translation, rotation)

        # Apply the transformation matrix to the sphere positions and port positions
        # Use the translation vector as the origin

        spheres_transformed = apply_transformation_matrix(translation.T,
                                                            sphere_positions.T,
                                                            t).T

        positions_transformed = apply_transformation_matrix(translation.T,
                                                            port_positions.T,
                                                            t).T

        return spheres_transformed, positions_transformed

    @staticmethod
    def _project(sphere_positions, sphere_radii, sample_points, sample_radii, element_bounds):
        pseudo_densities = calculate_pseudo_densities(sphere_positions, sphere_radii, sample_points, sample_radii,
                                                      element_bounds)
        return pseudo_densities


class Interconnect(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_segments', types=int)
        self.options.declare('radius', types=float)
        self.options.declare('color', types=str)

    def setup(self):

        # Unpack the options
        n_segments = self.options['n_segments']
        radius = self.options['radius']

        # Define the input shapes
        shape_control_points = (n_segments - 1, 3)
        shape_positions = (n_segments + 1, 3)
        shape_radii = (n_segments + 1, 1)

        # Define the inputs
        self.add_input('start_point', shape=(1, 3))
        self.add_input('control_points', shape=shape_control_points)
        self.add_input('end_point', shape=(1, 3))

        # Mesh Inputs
        self.add_input('element_length', val=0)
        self.add_input('element_bounds', shape_by_conn=True)
        self.add_input('centers', shape_by_conn=True)
        self.add_input('element_sphere_positions', shape_by_conn=True)
        self.add_input('element_sphere_radii', shape_by_conn=True)


        # Define the outputs
        self.add_output('transformed_sphere_positions', shape=shape_positions)
        self.add_output('transformed_sphere_radii', shape=shape_radii)
        # self.add_output('volume', val=0.0)

        # Outputs
        self.add_output('pseudo_densities',
                        compute_shape=lambda shapes: (shapes['centers'][0], shapes['centers'][1], shapes['centers'][2]))

    # def setup_partials(self):
    #     self.declare_partials('transformed_sphere_positions', ['start_point', 'control_points', 'end_point'])
    #     self.declare_partials('transformed_sphere_radii', ['start_point', 'control_points', 'end_point'])
    #     self.declare_partials('pseudo_densities', ['start_point', 'control_points', 'end_point'])

    def compute(self, inputs, outputs):

        # Unpack the inputs
        start_point = inputs['start_point']
        control_points = inputs['control_points']
        end_point = inputs['end_point']

        radius = self.options['radius']


        # vstack
        points = np.vstack([start_point, control_points, end_point])
        radii = radius * np.ones((points.shape[0], 1))

        # Calculate the positions
        # translated_positions = translate_linear_spline(sphere_positions, start_point, control_points, end_point)

        # Set the outputs
        outputs['transformed_sphere_positions'] = points
        outputs['transformed_sphere_radii'] = radii

        # Get the Mesh inputs
        element_length = inputs['element_length']
        element_bounds = inputs['element_bounds']
        sample_points = inputs['element_sphere_positions']
        sample_radii = inputs['element_sphere_radii']

        # Compute the pseudo-densities
        pseudo_densities = self._project(sample_points, sample_radii, points, radii)

        outputs['pseudo_densities'] = pseudo_densities

    @staticmethod
    def _project(sample_points, sample_radii, sphere_positions, sphere_radii):
        import numpy as np
        def create_cylinders(points, radius):
            x1 = np.array(points[:-1])  # Start positions (-1, 3)
            x2 = np.array(points[1:])  # Stop positions (-1, 3)
            r = np.full((x1.shape[0], 1), radius)
            return x1, x2, r

        # FIXME different radii for different int segments
        X1, X2, R = create_cylinders(sphere_positions, sphere_radii[0])

        pseudo_densities = calculate_combined_densities(sample_points, sample_radii, X1, X2, R)
        return pseudo_densities

    # def compute_partials(self, inputs, partials):
    #
    #     # Unpack the inputs
    #     start_point = inputs['start_point']
    #     control_points = inputs['control_points']
    #     end_point = inputs['end_point']
    #
    #     # Unpack the options
    #     radius = self.options['radius']
    #
    #     # Convert the inputs to Jax arrays
    #     start_point = jnp.array(start_point)
    #     control_points = jnp.array(control_points)
    #     end_point = jnp.array(end_point)
    #     positions = jnp.array(positions)
    #     radii = jnp.array(radii)
    #
    #     # Calculate the partial derivatives
    #     jac_translated_positions = jacfwd(translate_linear_spline, argnums=(1, 2, 3))
    #     jac_translated_positions_val = jac_translated_positions(positions, start_point, control_points, end_point)
    #
    #     # Slice the Jacobian
    #     jac_translated_positions_start_point = jac_translated_positions_val[0]
    #     jac_translated_positions_control_points = jac_translated_positions_val[1]
    #     jac_translated_positions_end_point = jac_translated_positions_val[2]
    #
    #     # Set the outputs
    #     partials['transformed_sphere_positions', 'start_point'] = jac_translated_positions_start_point
    #     partials['transformed_sphere_positions', 'control_points'] = jac_translated_positions_control_points
    #     partials['transformed_sphere_positions', 'end_point'] = jac_translated_positions_end_point



class System(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_projections', types=int, desc='Number of projections')
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

    def setup(self):
        # Get the options
        n_projections = self.options['n_projections']

        # Set the inputs
        self.add_input('element_length', val=0)

        for i in range(n_projections):
            self.add_input(f'pseudo_densities_{i}', shape_by_conn=True)

        # Set the outputs
        self.add_output('pseudo_densities', copy_shape='pseudo_densities_0')
        self.add_output('max_pseudo_density', val=0.0, desc='How much of each object overlaps/is out of bounds')
        # TODO output penalized and unpenalized, and min and w/o min

    def setup_partials(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the partials
        for i in range(n_projections):
            self.declare_partials('pseudo_densities', f'pseudo_densities_{i}')
            self.declare_partials('max_pseudo_density', f'pseudo_densities_{i}')


    def compute(self, inputs, outputs):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        element_length = inputs['element_length']
        pseudo_densities = [inputs[f'pseudo_densities_{i}'] for i in range(n_projections)]

        # Calculate the values
        aggregate_pseudo_densities, max_pseudo_density = self._aggregate_pseudo_densities(pseudo_densities, element_length, rho_min)


        # Write the outputs
        outputs['pseudo_densities'] = aggregate_pseudo_densities
        outputs['max_pseudo_density'] = max_pseudo_density

    def compute_partials(self, inputs, partials):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        element_length = np.array(inputs['element_length'])
        pseudo_densities = [np.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]

        # Calculate the partial derivatives
        jac_pseudo_densities, jac_max_pseudo_density = jacfwd(self._aggregate_pseudo_densities)(pseudo_densities, element_length, rho_min)

        # Set the partial derivatives
        jacs = zip(jac_pseudo_densities, jac_max_pseudo_density)
        for i, (jac_pseudo_densities_i, jac_max_pseudo_density_i) in enumerate(jacs):
            partials['pseudo_densities', f'pseudo_densities_{i}'] = jac_pseudo_densities_i
            partials['max_pseudo_density', f'pseudo_densities_{i}'] = jac_max_pseudo_density_i

    @staticmethod
    def _aggregate_pseudo_densities(pseudo_densities, element_length, rho_min):

        # Aggregate the pseudo-densities
        aggregate_pseudo_densities = np.zeros_like(pseudo_densities[0])
        for pseudo_density in pseudo_densities:
            aggregate_pseudo_densities += pseudo_density

        # Ensure that no pseudo-density is below the minimum value
        aggregate_pseudo_densities = np.maximum(aggregate_pseudo_densities, rho_min)

        # Calculate the maximum pseudo-density
        max_pseudo_density = kreisselmeier_steinhauser_max(aggregate_pseudo_densities)

        return aggregate_pseudo_densities, max_pseudo_density