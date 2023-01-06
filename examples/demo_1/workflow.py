"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson

"""

# Import packages
import os
import logging
import numpy as np
from SPI2Py.main import SPI2

# Set up logging
logging.basicConfig(filename='examples/demo_1/output.log', encoding='utf-8', level=logging.DEBUG)

# Get the system path for reading/writing example files
cwd = os.getcwd() + '/examples/demo_1/'

# SPI2 Workflow

# Initialize the class
demo = SPI2()


# Specify the input file
input_filepath = cwd + 'input.yaml'

demo.add_input_file(input_filepath)

# Specify the config file
config_filepath = cwd + 'config.yaml'
demo.add_configuration_file(config_filepath)

# Generate classes from the inputs file
demo.create_objects_from_input()

# Map the objects to a 3D layout
layout_generation_method = 'manual'
locations = np.array([-3., -4.41, -0.24, 0., 0., 0., 2., 4.41, 0.24, 0., 0., 0., -1., 2., 2.])
demo.generate_layout(layout_generation_method, inputs=locations)

# For development: Plot initial layout
demo.layout.plot_layout()

# Perform gradient-based optimization
demo.optimize_spatial_configuration()

# For development: Print Results
print('Result:', demo.result)

# For development: Plot the final layout to see the change
demo.layout.set_positions(demo.result.x)
demo.layout.plot_layout()

# Write output file
output_filepath = cwd + 'output.json'
demo.write_output(output_filepath)

