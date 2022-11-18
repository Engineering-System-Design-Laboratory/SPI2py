"""

TODO Add a logger
TODO place workflow in if name == __main__ statement
TODO Ensure add src to python path for pytest
"""

import numpy as np
import json
import yaml
from utils.layout_generator import generate_layout
from utils.spatial_topology import generate_random_layout
from utils.gradient_based_optimization import optimize
from utils.visualization import generate_gif
from datetime import datetime
from time import perf_counter_ns

'''Set the Filepaths'''


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_file = config['Inputs']['Folderpath'] + config['Inputs']['Filename']

with open(input_file, 'r') as f:
    inputs = yaml.safe_load(f)


'''Initialize the Layout'''


# Generate objects from the inputs file
layout = generate_layout(inputs)

# Generate a random initial layout
initial_layout_design_vector = generate_random_layout(layout)
layout.set_positions(initial_layout_design_vector)

# For development: Plot initial layout
layout.plot_layout()



'''Perform Gradient-Based Optimization'''


res, design_vector_log = optimize(layout)


'''Post Processing'''


# For development: Print Results
print('Result:', res)

# For development: Plot the final layout to see the change
layout.set_positions(res.x)
layout.plot_layout()


# Generate GIF
if config['Visualization']['Output GIF'] is True:
    generate_gif(layout, design_vector_log, 1, config['Outputs']['Folderpath'])


'''Write output file'''


# Create a timestamp
now = datetime.now()
now_formatted = now.strftime("%d/%m/%Y %H:%M:%S")

# TODO Create a prompt to ask user for comments on the results

# Create the output dictionary
outputs = {'Placeholder': 1,
           '':1,
           'Date and time': now_formatted,
           '':1,
           'Comments': 'Placeholder'}



output_file = config['Outputs']['Folderpath'] + config['Outputs']['Report Filename']
with open(output_file, 'w') as f:
    json.dump(outputs, f)
