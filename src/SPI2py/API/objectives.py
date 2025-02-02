import jax.numpy as jnp
from jax import jacrev
from openmdao.api import ExplicitComponent
from SPI2py.models.geometry.bounding_volume import bounding_box_volume, bounding_box_bounds


class BoundingBoxVolume(ExplicitComponent):

    def setup(self):
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)
        self.add_output('bounding_box_volume', shape=(1,))
        self.add_output('bounding_box_bounds', shape=(6,))

    def setup_partials(self):
        self.declare_partials('bounding_box_volume', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the input variables
        positions = jnp.array(inputs['sphere_positions'])
        radii = jnp.array(inputs['sphere_radii'])

        # Calculate the bounding box bounds and volume
        bb_volume, bb_bounds = self._bounding_box_volume(positions, radii)

        # Set the outputs
        outputs['bounding_box_bounds'] = bb_bounds
        outputs['bounding_box_volume'] = bb_volume

    def compute_partials(self, inputs, partials):

        # Get the input variables
        positions = jnp.array(inputs['sphere_positions'])
        radii = jnp.array(inputs['sphere_radii'])

        # Calculate the jacobian of the bounding box volume
        jac_bb_volume, _ = jacrev(self._bounding_box_volume)(positions, radii)

        # Set the outputs
        partials['bounding_box_volume', 'sphere_positions'] = jac_bb_volume

    @staticmethod
    def _bounding_box_volume(positions, radii):
        bb_bounds = bounding_box_bounds(positions, radii)
        bb_volume = bounding_box_volume(bb_bounds)
        return bb_volume, bb_bounds
