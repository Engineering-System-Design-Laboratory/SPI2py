import math
import numpy as np

class Fluid:
    def __init__(self, name, density, dynamic_viscosity):
        """
        Fluid properties class to define different fluids
        
        Args:
            name (str): Name of the fluid
            density (float): Density of the fluid in kg/m³
            dynamic_viscosity (float): Dynamic viscosity of the fluid in Pa·s
        """
        self.name = name
        self.density = density
        self.dynamic_viscosity = dynamic_viscosity

# Predefined fluids
WATER = Fluid("Water", density=998.2, dynamic_viscosity=0.001002)
AIR = Fluid("Air", density=1.225, dynamic_viscosity=1.81e-5)

def calculate_bend_angle(coord1, coord2, coord3,d,tolc=1e-6):
    """
    Calculate the angle between pipe segments at a bend. It is assumed that bend radius is 3 times the pipe diameter.
    
    Args:
        coord1 (array): First coordinate point
        coord2 (array): Bend coordinate point
        coord3 (array): Third coordinate point
    
    Returns:
        float: Angle between pipe segments in degrees
    """
    r=d*3
    # Create vectors
    vector1 = np.array(coord1) - np.array(coord2)
    vector2 = np.array(coord3) - np.array(coord2)
    
    # Normalize vectors
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    
    # Calculate angle using dot product
    cos_angle = np.dot(vector1_norm, vector2_norm)
    nu=(1-tolc)*cos_angle
    theta_radians = np.arccos(np.clip(nu, -1.0, 1.0))
    alpha=math.pi-theta_radians
    l=r*np.sqrt((1+nu)/(1-nu))
    
    
    return alpha,l

def calculate_pressure_drop(
    coordinates, 
    pipe_radius, 
    fluid=WATER, 
    flow_rate=None, 
):
    """
    Calculate pressure drop in a pipe with multiple segments and bends. It is assumed that the flow is incompressible and turbulent in a smooth pipe.
    
    Args:
        coordinates (list): List of coordinate tuples [(x1,y1,z1), (x2,y2,z2), ...]
        pipe_radius (float): Pipe radius in meters
        fluid (Fluid): Fluid object (default is water)
        flow_rate (float, optional): Volume flow rate in m³/s 
    
    Returns:
        float: total pressure drop 
    """
    # Validate input
    if len(coordinates) < 2:
        raise ValueError("At least two coordinates are required")
    
    # Convert coordinates to numpy array if it's a list
    if isinstance(coordinates, list):
        coordinates = np.array(coordinates)
    
    # Validate input
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("Coordinates must be a list or numpy array")
    
    # Total pressure drop accumulator
    total_pressure_drop = 0
    detailed_results = []
    
    # Pressure drop calculation for each pipe segment
    for i in range(len(coordinates) - 1):
        # Get start and end coordinates for this segment
        start_coords = coordinates[i]
        end_coords = coordinates[i+1]
        
        # Calculate pipe segment length
        segment_length = np.linalg.norm(
            np.array(end_coords) - np.array(start_coords)
        )
        
        # Calculate pipe diameter
        pipe_diameter = 2 * pipe_radius
        
        # If flow rate not provided, use a default assumption
        if flow_rate is None:
            # Typical flow velocity for water pipes (1-2 m/s)
            flow_velocity = 1.5  # m/s
            flow_rate = flow_velocity * math.pi * (pipe_radius**2)
        
        # Calculate flow velocity
        cross_sectional_area = math.pi * (pipe_radius**2)
        flow_velocity = flow_rate / cross_sectional_area
        
        # Calculate Reynolds number
        reynolds_number = (fluid.density * flow_velocity * pipe_diameter) / fluid.dynamic_viscosity
        if reynolds_number < 2100:
            raise ValueError("Flow is not turbulent. Reynolds number must be greater than 2100 for this calculation.")
        
        # Estimate friction factor (Blasius correlation)
        friction_factor = 0.316/ reynolds_number**0.25
        # Darcy-Weisbach equation constants
        gravity = 9.81  # m/s²
        
        # Major losses (friction)
        major_loss_head = (
            friction_factor * 
            (segment_length / pipe_diameter) * 
            (flow_velocity**2 / (2 * gravity))
        )
        
        # Calculate bend loss if not the last segment
        minor_loss_head = 0
        bend_angle = 0
        if i < len(coordinates) - 2:
            # Calculate bend angle
            alpha, l_c = calculate_bend_angle(
                coordinates[i], 
                coordinates[i+1], 
                coordinates[i+2],
                pipe_diameter
            )
            
            # Estimate minor loss coefficient for bend
            # This is a simplified model and can be refined
            bend_radius_ratio = 3  # Bend radius is 3 times the pipe diameter
            K_bend = (friction_factor * alpha*bend_radius_ratio)+(0.1+2.4*friction_factor)*np.sin(alpha/2)+(6.6*friction_factor*(np.sin(alpha/2)+np.sqrt(np.sin(alpha/2)+1e-6)))/((bend_radius_ratio)**(4*alpha/math.pi))
            minor_loss_head = (
                K_bend * 
                (flow_velocity**2 / (2 * gravity))
            )
        
        # Total head loss for this segment
        segment_head_loss = major_loss_head + minor_loss_head
        
        # Pressure drop for this segment
        segment_pressure_drop = (
            segment_head_loss * 
            fluid.density * 
            gravity
        )
        
        # Accumulate total pressure drop
        total_pressure_drop += segment_pressure_drop
        
        # Store detailed results for this segment
        detailed_results.append({
            'segment': i+1,
            'length': segment_length,
            'flow_velocity': flow_velocity,
            'reynolds_number': reynolds_number,
            'friction_factor': friction_factor,
            'major_head_loss': major_loss_head,
            'minor_head_loss': minor_loss_head,
            'bend_angle': bend_angle,
            'segment_pressure_drop': segment_pressure_drop
        })
    
    return total_pressure_drop
    #{
        #'total_pressure_drop': total_pressure_drop,
        #'fluid': fluid.name,
        #'segments': detailed_results
    #}
