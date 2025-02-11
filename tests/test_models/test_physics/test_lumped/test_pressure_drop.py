import unittest
import numpy as np
from SPI2py.models.physics.lumped.pressure_drop import calculate_pressure_drop, Fluid, WATER, AIR

class TestPressureDropCalculation(unittest.TestCase):
    def setUp(self):
        # Standard test parameters
        self.test_radius = 0.05  # 5 cm pipe radius
        self.test_flow_rate = 0.001  # 1 L/s
    
    def test_straight_horizontal_pipe(self):
        """Test pressure drop in a straight horizontal pipe"""
        coordinates = [
            (0, 0, 0),
            (10, 0, 0)  # 10m long horizontal pipe
        ]
        
        pressure_drop = calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=self.test_radius,
            flow_rate=self.test_flow_rate
        )
        
        self.assertIsInstance(pressure_drop, float)
        self.assertGreater(pressure_drop, 0)

    def test_pipe_with_0_degree_bend(self):
        """Test pressure drop in a straight horizontal pipe"""
        coordinates1 = [
            (0, 0, 0),
            (0, 10, 0),
            (0,20,0)  # 20m long horizontal pipe
        ]
        
        pressure_drop1 = calculate_pressure_drop(
            coordinates=coordinates1,
            pipe_radius=self.test_radius,
            flow_rate=self.test_flow_rate
        )
        coordinates2 = [
            (0, 0, 0),
            (0,20,0)  # 20m long horizontal pipe
        ]
        
        pressure_drop2 = calculate_pressure_drop(
            coordinates=coordinates1,
            pipe_radius=self.test_radius,
            flow_rate=self.test_flow_rate
        )
        
        self.assertIsInstance(pressure_drop1, float)
        self.assertGreater(pressure_drop1, 0)
        self.assertIsInstance(pressure_drop2, float)
        self.assertGreater(pressure_drop2, 0)
        self.assertAlmostEqual(pressure_drop1, pressure_drop2, delta=1)
    
    def test_pipe_with_90_degree_bend(self):
        """Test pressure drop in a pipe with a 90-degree bend"""
        coordinates = [
            (0, 0, 0),
            (10, 0, 0),
            (10, 10, 0)
        ]
        
        pressure_drop = calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=self.test_radius,
            flow_rate=self.test_flow_rate
        )
        
        self.assertIsInstance(pressure_drop, float)
        self.assertGreater(pressure_drop, 0)
    
    def test_numpy_array_input(self):
        """Test pressure drop calculation with numpy array input"""
        coordinates = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0]
        ])
        
        pressure_drop = calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=self.test_radius,
            flow_rate=self.test_flow_rate
        )
        
        self.assertIsInstance(pressure_drop, float)
        self.assertGreater(pressure_drop, 0)
    
    def test_different_fluids(self):
        """Test pressure drop calculation with different fluids"""
        coordinates = [
            (0, 0, 0),
            (10, 0, 0)
        ]
        
        water_drop = calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=self.test_radius,
            fluid=WATER,
            flow_rate=self.test_flow_rate
        )
        
        air_drop = calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=self.test_radius,
            fluid=AIR,
            flow_rate=self.test_flow_rate
        )
        
        self.assertNotAlmostEqual(water_drop, air_drop, delta=1)
    
    def test_custom_fluid(self):
        """Test pressure drop calculation with custom fluid"""
        custom_fluid = Fluid(
            name="Custom Fluid",
            density=1000,
            dynamic_viscosity=0.001
        )
        
        coordinates = [
            (0, 0, 0),
            (10, 0, 0)
        ]
        
        pressure_drop = calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=self.test_radius,
            fluid=custom_fluid,
            flow_rate=self.test_flow_rate
        )
        
        self.assertIsInstance(pressure_drop, float)
        self.assertGreater(pressure_drop, 0)
    
    def test_laminar_flow_error(self):
        """Test if error is raised for laminar flow"""
        coordinates = [
            (0, 0, 0),
            (10, 0, 0)
        ]
        
        # Very low flow rate to ensure laminar flow
        with self.assertRaises(ValueError):
            calculate_pressure_drop(
                coordinates=coordinates,
                pipe_radius=0.1,  # Large radius
                flow_rate=1e-6    # Very low flow rate
            )
    
    def test_insufficient_coordinates(self):
        """Test if error is raised for insufficient coordinates"""
        coordinates = [(0, 0, 0)]
        
        with self.assertRaises(ValueError):
            calculate_pressure_drop(
                coordinates=coordinates,
                pipe_radius=self.test_radius
            )
    
    def test_invalid_input_type(self):
        """Test if error is raised for invalid input type"""
        coordinates = "invalid input"
        
        with self.assertRaises(TypeError):
            calculate_pressure_drop(
                coordinates=coordinates,
                pipe_radius=self.test_radius
            )
