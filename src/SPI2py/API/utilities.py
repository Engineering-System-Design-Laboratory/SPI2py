import jax.numpy as jnp
from jax import jacfwd, jacrev
from openmdao.core.explicitcomponent import ExplicitComponent
from ..models.utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min
from ..models.utilities.input_and_output import read_input_file


class Multiplexer(ExplicitComponent):
    """
    An ExplicitComponent that vertically stacks a series of inputs with sizes (n, 3) or (n, 1).
    """

    def initialize(self):
        # Initialize with a list of sizes for each input
        self.options.declare('n_i', types=list, desc='The number of spheres for each component')
        self.options.declare('m', types=int, desc='Column size, either 1 or 3')

    def setup(self):
        n_i = self.options['n_i']
        n = sum(n_i)
        m = self.options['m']

        # Define inputs and output
        for i, size in enumerate(n_i):
            self.add_input(f'input_{i}', shape=(size, m))

        self.add_output('stacked_output', shape=(n, m))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the options
        n_i = self.options['n_i']
        m = self.options['m']  # FOR DEBUGGING

        # Get the input arrays
        input_arrays = ()
        for i in range(len(n_i)):
            input_arrays = input_arrays + (inputs[f'input_{i}'],)

        # Convert the input arrays to torch tensors
        input_tensors = ()
        for input_array in input_arrays:
            input_tensors = input_tensors + (jnp.array(input_array),)

        # Stack inputs vertically
        stacked_output = self._multiplex(*input_tensors)

        # Set the output
        outputs['stacked_output'] = stacked_output

    def compute_partials(self, inputs, partials):

        # Get the options
        n_i = self.options['n_i']

        # Get the input arrays
        input_arrays = ()
        for i in range(len(n_i)):
            input_arrays = input_arrays + (inputs[f'input_{i}'],)

        # Convert the input arrays to torch tensors
        input_tensors = ()
        for input_array in input_arrays:
            input_tensors = input_tensors + (jnp.array(input_array),)

        # Calculate the partial derivatives wrt all inputs
        argnums = tuple(range(len(n_i)))
        jac_stacked_output = jacfwd(self._multiplex, argnums=argnums)(*input_tensors)

        # Convert the partial derivatives to numpy arrays
        jac_stacked_output_np = []
        for jac in jac_stacked_output:
            jac_stacked_output_np.append(jac)

        # Set the partial derivatives
        for i in range(len(n_i)):
            partials['stacked_output', f'input_{i}'] = jac_stacked_output_np[i]


    @staticmethod
    def _multiplex(*args):
        return jnp.vstack(args)



class Aggregator(ExplicitComponent):
    """
    Takes an array and aggregates it into a single value.
    """

    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('m', types=int)

    def setup(self):

        n = self.options['n']
        m = self.options['m']

        # Define inputs and output
        self.add_input('input_vector', shape=(n, m))
        self.add_output('aggregated_output', shape=(1, 1))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the input arrays
        input_vector = inputs['input_vector']

        # Convert the input arrays to torch tensors
        input_vector = jnp.array(input_vector)

        # Stack inputs vertically
        aggregated_output = self._compute_aggregation(input_vector)

        # Convert the stacked output to a numpy array
        aggregated_output = aggregated_output.detach().numpy()

        # Set the output
        outputs['aggregated_output'] = aggregated_output

    def compute_partials(self, inputs, partials):
        # Get the input arrays
        input_vector = inputs['input_vector']

        # Convert the input arrays to torch tensors
        input_vector = jnp.array(input_vector)

        # Calculate the partial derivatives
        jac_aggregated_output = jacrev(self._compute_aggregation)(input_vector)

        # Convert the partial derivatives to numpy arrays
        jac_aggregated_output_np = jac_aggregated_output.detach().numpy()

        # Set the partial derivatives
        partials['aggregated_output', 'input_vector'] = jac_aggregated_output_np

    @staticmethod
    def _compute_aggregation(input_vector):
        raise NotImplementedError


class MaxAggregator(Aggregator):

    @staticmethod
    def _compute_aggregation(input_vector):
        aggregated_output = kreisselmeier_steinhauser_max(input_vector)
        return aggregated_output


class MinAggregator(Aggregator):

    @staticmethod
    def _compute_aggregation(input_vector):
        aggregated_output = kreisselmeier_steinhauser_min(input_vector)
        return aggregated_output
