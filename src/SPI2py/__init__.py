# from SPI2py.group_model.component_kinematics.kinematics import KinematicsInterface
# from SPI2py.group_model.OpenMDAO_Objects.Interconnects import Interconnect
# from SPI2py.group_model.OpenMDAO_Objects.Components import Component
# from SPI2py.group_model.OpenMDAO_Objects.Systems import System

# Configure JAX to use 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)
