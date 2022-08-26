"""Module...
Docstring

"""


import numpy as np
import networkx as nx
from itertools import product, combinations
from utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms


class Object:

    # Can I add a component instance counter here? Do I need it?

    def __init__(self, name, positions, radii):
        self.name = name
        self.positions = positions
        self.radii = radii






class Component(Object):

    # def __init__(self):
    #     pass

    def get_design_vector(self):
        pass

class Interconnect(Object):
    def __init__(self):
        pass

class InterconnectNode(Object):
    def __init__(self):
        pass

class Structure(Object):
    def __init__(self):
        pass


class Layout:
    def __init__(self):
        pass

        # Objective pairs

        # Collision pairs

