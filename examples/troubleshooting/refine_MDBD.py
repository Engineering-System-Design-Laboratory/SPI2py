from jax import numpy as jnp, grad, jacfwd
from scipy.optimize import minimize


def refine_mdbd(xyzr_0):


    def objective(x):
        """Minimize the volume of the spheres"""
        x = x.reshape(-1, 4)
        radii = x[:, 3]
        volumes = 4/3 * jnp.pi * radii**3
        volume = jnp.sum(volumes)

        return -volume

    def constraint_1(x):
        """No overlap between spheres"""
        x = x.reshape(-1, 4)
        centers = x[:, :3]
        radii = x[:, 3]

        centers_a = centers.reshape(-1, 1, 3)
        centers_b = centers.reshape(1, -1, 3)
        radii_a = radii.reshape(-1, 1)
        radii_b = radii.reshape(1, -1)

        d = jnp.linalg.norm(centers_a - centers_b, axis=2)

        triu_indices = jnp.triu_indices(len(centers_a), k=1)

        d = d[triu_indices]
        radii_a = radii_a.reshape(-1)[triu_indices[0]]
        radii_b = radii_b.reshape(-1)[triu_indices[1]]

        signed_distances = d - (radii_a + radii_b)

        c = signed_distances

        # c = jnp.min(c)

        return c

    def constraint_2(x):
        """Stay within the bounds of the cube"""
        x = x.reshape(-1, 4)
        positions = x[:, :3]
        radii = x[:, 3]

        lower_bound = 0
        upper_bound = 1
        tolerance = 1e-6

        # Lower bounds: positions - radii >= lower_bound
        c_lower = (positions - radii[:, jnp.newaxis]) - lower_bound + tolerance

        # Upper bounds: positions + radii <= upper_bound
        c_upper = upper_bound - (positions + radii[:, jnp.newaxis]) + tolerance

        # Combine all constraints
        c = jnp.concatenate((c_lower.flatten(), c_upper.flatten()))

        # c = jnp.min(c)

        return c  # Should be >= 0


    grad_f = grad(objective)
    # grad_c1 = grad(constraint_1)
    # grad_c2 = grad(constraint_2)
    grad_c1 = jacfwd(constraint_1)
    grad_c2 = jacfwd(constraint_2)

    # Analyze initial guess
    f_0 = objective(xyzr_0)
    c1_0 = constraint_1(xyzr_0)
    c2_0 = constraint_2(xyzr_0)

    print(f'Initial guess: Volume Fraction = {f_0}')
    # print(f'Max overlap = {c1_0}')
    # print(f'Bounds violation = {c2_0}')
    print(' ')

    # Run the optimization
    res = minimize(objective, xyzr_0,
                   jac=grad_f,
                   tol=1e-6,
                   constraints=[{'type': 'ineq', 'fun': constraint_1, 'jac':grad_c1},
                                {'type': 'ineq', 'fun': constraint_2, 'jac':grad_c2}])

    # Analyze the result
    f_res = objective(res.x)
    c1_res = constraint_1(res.x)
    c2_res = constraint_2(res.x)

    print(f'Result: Volume Fraction = {f_res}')
    # print(f'Max overlap = {c1_res}')
    # print(f'Bounds violation = {c2_res}')

    return res

# xyzr_0 = np.hstack((sphere_points, 0.5*sphere_radii)).flatten()
#
# res = refine_mdbd(xyzr_0)
