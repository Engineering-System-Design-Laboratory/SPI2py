import jax.numpy as jnp
from SPI2py.models.geometry.encapsulation import overlap_volume_sphere_sphere

def test_perfect_overlap():
    r_1 = jnp.array([5.0])
    r_2 = jnp.array([5.0])
    d = jnp.array([0.0])
    expected = (4 / 3) * jnp.pi * r_1 ** 3
    result = overlap_volume_sphere_sphere(r_1, r_2, d)
    assert jnp.allclose(result, expected)

def test_no_overlap():
    r_1 = jnp.array([5.0])
    r_2 = jnp.array([5.0])
    d = jnp.array([10.0])
    expected = jnp.array([0.0])
    result = overlap_volume_sphere_sphere(r_1, r_2, d)
    assert jnp.allclose(result, expected)

def test_no_overlap_2():
    r_1 = jnp.array([5.0])
    r_2 = jnp.array([5.0])
    d = jnp.array([12.0])
    expected = jnp.array([0.0])
    result = overlap_volume_sphere_sphere(r_1, r_2, d)
    assert jnp.allclose(result, expected)

def test_almost_no_overlap():
    r_1 = jnp.array([5.0])
    r_2 = jnp.array([5.0])
    d = jnp.array([0.001])  # A small distance should not result in "almost infinity"
    expected = jnp.array([523.6])  # Per http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm
    result = overlap_volume_sphere_sphere(r_1, r_2, d)
    expected_rounded = jnp.round(expected, decimals=1)
    result_rounded = jnp.round(result, decimals=1)
    assert jnp.allclose(result_rounded, expected_rounded)

def test_partial_overlap():
    r_1 = jnp.array([5.0])
    r_2 = jnp.array([5.0])
    d = jnp.array([5.0])
    expected = jnp.array(163.62)  # Per http://ambrnet.com/TrigoCalc/Sphere/TwoSpheres/Intersection.htm
    result = overlap_volume_sphere_sphere(r_1, r_2, d)
    result = jnp.round(result, decimals=2)
    assert jnp.allclose(result, expected)
