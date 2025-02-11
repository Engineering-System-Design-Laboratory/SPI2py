from SPI2py.models.physics.distributed import assembly as asb
from SPI2py.models.physics.lumped import pressure_drop as pd
import numpy as np
import matplotlib.pyplot as plt
def main():
    # Example 1: Simple straight pipe
    print("Example 1: Straight pipe")
    coordinates = [
        (0, 0, 0),
        (10, 0, 0)
    ]
    
    pressure_drop = pd.calculate_pressure_drop(
        coordinates=coordinates,
        pipe_radius=0.05,  # 5 cm radius
        fluid=pd.WATER,
        flow_rate=0.001    # 1 L/s
    )
    
    print(f"Pressure drop in straight pipe: {pressure_drop:.2f} Pa")
    
    # Example 2: Pipe with multiple bends
    print("\nExample 2: Pipe with multiple bends")
    coordinates = [
        (0, 0, 0),      # Start
        (5, 0, 0),      # First segment
        (5, 3, 0),      # 90-degree bend
        (8, 3, 0),      # Second segment
        (8, 3, 2)       # Final vertical segment
    ]
    
    pressure_drop = pd.calculate_pressure_drop(
        coordinates=coordinates,
        pipe_radius=0.025,  # 2.5 cm radius
        fluid=pd.WATER,
        flow_rate=0.0005   # 0.5 L/s
    )
    
    print(f"Pressure drop in bent pipe: {pressure_drop:.2f} Pa")
    
    # Example 3: Comparing different fluids
    print("\nExample 3: Comparing water and air")
    coordinates = [
        (0, 0, 0),
        (5, 0, 0),
        (5, 5, 0)
    ]
    
    water_drop = pd.calculate_pressure_drop(
        coordinates=coordinates,
        pipe_radius=0.05,
        fluid=pd.WATER,
        flow_rate=0.01
    )
    
    air_drop = pd.calculate_pressure_drop(
        coordinates=coordinates,
        pipe_radius=0.05,
        fluid=pd.AIR,
        flow_rate=0.01
    )
    
    print(f"Pressure drop with water: {water_drop:.2f} Pa")
    print(f"Pressure drop with air: {air_drop:.2f} Pa")
    
    # Example 4: Custom fluid (e.g., oil)
    print("\nExample 4: Custom fluid (oil)")
    oil = pd.Fluid(
        name="Oil",
        density=900,              # kg/m³
        dynamic_viscosity=0.03    # Pa·s
    )
    
    pressure_drop = pd.calculate_pressure_drop(
        coordinates=coordinates,
        pipe_radius=0.05,
        fluid=oil,
        flow_rate=0.01
    )
    
    print(f"Pressure drop with oil: {pressure_drop:.2f} Pa")
    
    # Example 5: Flow rate study
    print("\nExample 5: Flow rate study")
    flow_rates = np.linspace(0.001, 0.01, 10)  # 0.5 to 2 L/s
    pressure_drops = []
    
    for flow_rate in flow_rates:
        pressure_drop = pd.calculate_pressure_drop(
            coordinates=coordinates,
            pipe_radius=0.05,
            fluid=pd.WATER,
            flow_rate=flow_rate
        )
        pressure_drops.append(pressure_drop)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(flow_rates * 1000, pressure_drops, 'b-o')
    plt.xlabel('Flow Rate (L/s)')
    plt.ylabel('Pressure Drop (Pa)')
    plt.title('Pressure Drop vs Flow Rate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()