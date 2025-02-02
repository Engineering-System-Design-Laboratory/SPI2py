import jax.numpy as jnp
from jax import jacfwd
from openmdao.api import ExplicitComponent, Group

from ..models.computational_geometry.distance_calculations import signed_distances_spheres_spheres


# class CombinatorialCollisionDetection(Group):
#     raise NotImplementedError


# class PairwiseCollisionDetection(ExplicitComponent):
#
#     def initialize(self):
#         self.options.declare('n_spheres', types=int, desc='Number of spheres for input a')
#         self.options.declare('m_spheres', types=int, desc='Number of spheres for input b')
#
#     def setup(self):
#         n = self.options['n_spheres']
#         m = self.options['m_spheres']
#
#         self.add_input('positions_a', shape=(n, 3))
#         self.add_input('radii_a', shape=(n, 1))
#
#         self.add_input('positions_b', shape=(m, 3))
#         self.add_input('radii_b', shape=(m, 1))
#
#         self.add_output('signed_distances', shape=(n, m))
#
#     def setup_partials(self):
#         self.declare_partials('signed_distances', 'positions_a')
#         self.declare_partials('signed_distances', 'radii_a')
#         self.declare_partials('signed_distances', 'positions_b')
#         self.declare_partials('signed_distances', 'radii_b')
#
#     def compute(self, inputs, outputs):
#
#         # Extract the inputs
#         positions_a = inputs['positions_a']
#         radii_a = inputs['radii_a']
#         positions_b = inputs['positions_b']
#         radii_b = inputs['radii_b']
#
#         # Convert the inputs to torch tensors
#         positions_a = torch.tensor(positions_a, dtype=torch.float64)
#         radii_a = torch.tensor(radii_a, dtype=torch.float64)
#         positions_b = torch.tensor(positions_b, dtype=torch.float64)
#         radii_b = torch.tensor(radii_b, dtype=torch.float64)
#
#         # Calculate the signed distances
#         signed_distances = self._compute_signed_distances(positions_a, radii_a, positions_b, radii_b)
#
#         # Convert the outputs to numpy arrays
#         signed_distances = signed_distances.detach().numpy()
#
#         # Write the outputs
#         outputs['signed_distances'] = signed_distances
#
#     def compute_partials(self, inputs, partials):
#
#         # Extract the inputs
#         positions_a = inputs['positions_a']
#         radii_a = inputs['radii_a']
#         positions_b = inputs['positions_b']
#         radii_b = inputs['radii_b']
#
#         # Convert the inputs to torch tensors
#         positions_a = torch.tensor(positions_a, dtype=torch.float64, requires_grad=True)
#         radii_a = torch.tensor(radii_a, dtype=torch.float64, requires_grad=True)
#         positions_b = torch.tensor(positions_b, dtype=torch.float64, requires_grad=True)
#         radii_b = torch.tensor(radii_b, dtype=torch.float64, requires_grad=True)
#
#         # Define the Jacobian matrices using PyTorch Autograd
#         jac_signed_distances = jacfwd(self._compute_signed_distances, argnums=(0, 1, 2, 3))
#
#         # Evaluate the Jacobian matrices
#         jac_signed_distances_vals = jac_signed_distances(positions_a, radii_a, positions_b, radii_b)
#
#         # Slice the Jacobian
#         jac_signed_distances_positions_a = jac_signed_distances_vals[0].detach().numpy()
#         jac_signed_distances_radii_a = jac_signed_distances_vals[1].detach().numpy()
#         jac_signed_distances_positions_b = jac_signed_distances_vals[2].detach().numpy()
#         jac_signed_distances_radii_b = jac_signed_distances_vals[3].detach().numpy()
#
#         # Write the outputs
#         partials['signed_distances', 'positions_a'] = jac_signed_distances_positions_a
#         partials['signed_distances', 'radii_a'] = jac_signed_distances_radii_a
#         partials['signed_distances', 'positions_b'] = jac_signed_distances_positions_b
#         partials['signed_distances', 'radii_b'] = jac_signed_distances_radii_b
#
#     @staticmethod
#     def _compute_signed_distances(positions_a, radii_a, positions_b, radii_b):
#         signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b)
#         return signed_distances

