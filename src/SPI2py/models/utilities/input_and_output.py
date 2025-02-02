import tomli

def read_input_file(input_file_path):
    with open(input_file_path, 'rb') as f:
        input_file = tomli.load(f)

    return input_file


def read_xyzr_file(filepath, num_spheres=100):
    """
    Reads a .xyzr file and returns the positions and radii of the spheres.

    TODO Remove num spheres

    :param filepath:
    :param num_spheres:
    :return: positions, radii
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if num_spheres is not None and num_spheres > len(lines):
        raise ValueError('num_spheres must be less than the number of spheres in the file')

    # Truncate the number of spheres as specified
    lines = lines[0:num_spheres]

    positions = []
    radii = []

    for line in lines:
        x, y, z, r = line.split()

        positions.append([float(x), float(y), float(z)])
        radii.append(float(r))

    return positions, radii
