import numpy as np
import torch
import pyvista as pv
from SPI2py.models.computational_geometry.distance_calculations import percentage_overlap

# # Example: Cube center at (0, 0, 0), side length 2, recursion level 1
# cube_center = (0, 0, 0)
# cube_side = 2
# recursion_level = 1
#
# spheres = generate_spheres_for_cube(cube_center, cube_side, recursion_level)
# for sphere in spheres:
#     print(f"Sphere Center: ({sphere[0]}, {sphere[1]}, {sphere[2]}), Radius: {sphere[3]}")

# Define mesh grid parameters
delta_xyz = 0.1
n_elements_x = 4
n_elements_y = 4
n_elements_z = 4
element_radius = delta_xyz / 2
rho_0 = 0.01  # Nonzero default density for the mesh grid

# Create the mesh grid
x = np.arange(0, n_elements_x * delta_xyz, delta_xyz)
y = np.arange(0, n_elements_y * delta_xyz, delta_xyz)
z = np.arange(0, n_elements_z * delta_xyz, delta_xyz)
X, Y, Z = np.meshgrid(x, y, z)

rho_0_meshgrid = np.ones(X.shape) * rho_0
rho_meshgrid = np.zeros(X.shape)

# Define an object to project
sphere_positions = np.array([[0.15, 0.15, 0.15], [0.25, 0.25, 0.25], [0.35, 0.35, 0.35]])
sphere_radii = np.array([[0.05], [0.05], [0.05]])

# positions_1 = torch.tensor([[0., 0., 0.]])
# radii_1 = torch.tensor([[1]])
# # positions_2 = torch.tensor([[2, 2, 2]])
# # positions_2 = torch.tensor([[0., 0., 0.]])
# positions_2 = torch.tensor([[0.5, 0.5, 0.5]])
# radii_2 = torch.tensor([[1]])

positions_1 = torch.tensor(sphere_positions)
radii_1 = torch.tensor(sphere_radii)

# Convert meshgrid to tensor
positions_2 = torch.tensor(np.array([X.flatten(), Y.flatten(), Z.flatten()]).T)
radii_2 = torch.tensor(np.ones(X.shape).flatten() * element_radius)

overlap = percentage_overlap(positions_1, radii_1, positions_2, radii_2)


# Plot the mesh grid and the object
plotter = pv.Plotter()

# Create a MultiBlock for the mesh grid spheres
grid_spheres = pv.MultiBlock()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            sphere = pv.Sphere(center=[X[i, j, k], Y[i, j, k], Z[i, j, k]], radius=element_radius)
            grid_spheres.append(sphere)

# Add the grid_spheres MultiBlock to the plotter
plotter.add_mesh(grid_spheres, color='white', opacity=0.1)

for i in range(sphere_positions.shape[0]):
    plotter.add_mesh(pv.Sphere(center=sphere_positions[i], radius=sphere_radii[i]), color='blue')

plotter.show()







# rho_min = 1e-3;
# rho_d = torch.zeros((constants['num_models'], nelx * nely * nelz)).to(device)
# rho = torch.zeros((nelx * nely * nelz)).to(device)
#
# for i in range(0, constants['num_models']):
#     sph_inds = torch.arange(2 * i * constants['num_sph'],
#                             (2 * i + 1) * constants['num_sph'])  ## index of spheres for each component
#
#     radii = torch.tile(mdbds[sph_inds, -1].reshape((-1, 1)), (1, MESH['centers'].shape[0])).reshape(
#         (-1, 1))  # Radius of circle
#     x0v = torch.tile(cur_pos[sph_inds, :3], (1, MESH['centers'].shape[0])).reshape(
#         (-1, 3))  # The current location of each component
#     mesh_cent = torch.tile(MESH['centers'], (constants['num_sph'], 1))  # Center of each mesh
#     dd = torch.sqrt(torch.sum((mesh_cent - x0v) ** 2, axis=1)).reshape(
#         (-1, 1))  # Distance between the center of spheres and meshes
#     phi_q = dd - radii  # Distance between each mesh and sphere
#     phi_d = (1 - rho_min) / (1 + torch.exp(20 * phi_q)) + rho_min  # Map the distance to [0,1] value for meshes
#     rho_x = torch.reshape(phi_d, (constants['num_sph'], nelx * nely * nelz))
#     rho_d[i, :] = torch.sum(rho_x, axis=0)
#     rho += rho_d[i, :]
# rho = (1 - rho_min) / (1 + torch.exp(-1 * (rho - 0.5))) + rho_min
# rho_d = (1 - rho_min) / (1 + torch.exp(-1 * (rho_d - 0.5))) + rho_min
# rho[rho > 1] = 1
# rho_d[rho_d > 1] = 1




# Corey's code
# class GeomProj(nn.Module):
#     """
#     mesh
#
#     """
#
#     def __init__(self, L_x, L_y, nelx, nely, centers, bar_nodes, widths, edges, dev_nodes):
#         super(GeomProj, self).__init__()
#         self.L_x = L_x
#         self.L_y = L_y
#         self.nelx = nelx
#         self.nely = nely
#         self.centers = centers
#         self.bar_nodes = bar_nodes
#         self.widths = widths
#         self.edges = edges
#         self.dev_nodes = dev_nodes
#         self.nEl = nelx * nely
#         self.ndev = 3
#         self.nbars = 12
#         self.rho_min = 1e-3
#         self.rad = (torch.sqrt(torch.tensor(2)) * (L_x / nelx) / 2)
#         self.p = 8
#
#     def kron(self, a, b):
#         siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
#         res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
#         siz0 = res.shape[:-4]
#         out = res.reshape(siz0 + siz1)
#         return out
#
#     def projectBar(self, nodes, width, mesh_centers):
#         # calculate projection of the bar onto the mesh and derivative
#         # of the projection
#         w_max = 0.02
#         w = w_max * width
#         x0 = nodes[0, :]
#         xf = nodes[1, :]
#
#         xfv = xf.repeat(mesh_centers.size(0), 1)
#         x0v = x0.repeat(mesh_centers.size(0), 1)
#
#         # preprocess
#         a = xfv - x0v
#         b = mesh_centers - x0v
#         e = mesh_centers - xfv
#
#         # nan issues
#         a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
#         b = torch.where(torch.isnan(b), torch.zeros_like(b), b)
#         e = torch.where(torch.isnan(e), torch.zeros_like(e), e)
#
#         ghat = torch.stack([-a[:, 1], a[:, 0]], dim=1)
#         g = (torch.sum(ghat * b, dim=1) / torch.norm(ghat, dim=1, p=2).pow(2)).unsqueeze(1) * ghat
#         # g[torch.isnan(g)] = 0  # if x0&xf overlap then g = 0
#         g = torch.where(torch.isnan(g), torch.zeros_like(g), g)
#
#         aNorm = torch.norm(a, dim=1)
#         bNorm = torch.norm(b, dim=1)
#         eNorm = torch.norm(e, dim=1)
#         gNorm = torch.norm(g, dim=1)
#
#         # Find min distance
#         idxP1 = torch.nonzero(torch.sum(a * b, dim=1) <= 0).squeeze()
#         idxP2 = torch.nonzero((torch.sum(a * b, dim=1) > 0) * (torch.sum(a * b, dim=1) < torch.sum(a * a, dim=1)) * (
#                     torch.sum(a * a, dim=1) != 0)).squeeze()
#         idxP3 = torch.nonzero(
#             (torch.sum(a * b, dim=1) >= torch.sum(a * a, dim=1)) * (torch.sum(a * a, dim=1) != 0)).squeeze()
#         d = torch.zeros(self.nEl)
#         d[idxP1] = bNorm[idxP1]
#         d[idxP2] = gNorm[idxP2]
#         d[idxP3] = eNorm[idxP3]
#
#         # left here
#         # Calculate partial boundaries
#         phi_q = d - w / 2
#         idxR1 = torch.nonzero(phi_q > self.rad).squeeze()  # phi > r
#         idxR2 = torch.nonzero(torch.abs(phi_q) <= self.rad).squeeze()  # -r <= phi <= r
#         idxR3 = torch.nonzero(phi_q < -self.rad).squeeze()  # phi < -r
#         rho_q = torch.zeros(self.nEl)
#
#         # rho_q[idxR1] = torch.zeros_like(idxR1).float()
#         # rho_q[idxR1] = torch.zeros_like(idxR1)
#         # rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (self.rad ** 2 * torch.acos(phi_q[idxR2] / self.rad) - phi_q[idxR2] * torch.sqrt(self.rad ** 2 - phi_q[idxR2] ** 2))
#
#         # sqrt_tol = 1e-6
#         # tol_tens = torch.full_like(phi_q[idxR2], sqrt_tol)
#
#         # rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (self.rad ** 2 * torch.acos(phi_q[idxR2] / self.rad) - phi_q[idxR2] * torch.sqrt(torch.maximum((self.rad ** 2 - phi_q[idxR2] ** 2), tol_tens)))
#
#         # prevent nan values (instabilities during training)
#         input_acos = torch.clamp(phi_q[idxR2] / self.rad, min=-1, max=1)
#         input_sqrt = torch.clamp(self.rad ** 2 - phi_q[idxR2] ** 2, min=0.0)
#         rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (
#                     self.rad ** 2 * torch.acos(input_acos) - phi_q[idxR2] * torch.sqrt(input_sqrt))
#
#         # rho_q[idxR3] = torch.ones_like(idxR3).float()
#         rho_q[idxR3] = 1
#         # rho_q[idxR3] = torch.ones_like(idxR3)
#         # Convert to density
#         rho_tilde = self.rho_min + rho_q * (1 - self.rho_min)
#         rho_hat = rho_tilde
#
#         return rho_hat
#
#     def transform_rot(self, rot_theta, center, x, y):
#         # rotation about point (center of components)
#         # center point
#         x0 = center[0]
#         y0 = center[1]
#         # rotation
#         xrot = (x - x0) * torch.cos(rot_theta) - (y - y0) * torch.sin(rot_theta) + x0
#         yrot = (x - x0) * torch.sin(rot_theta) + (y - y0) * torch.cos(rot_theta) + y0
#
#         return xrot, yrot
#
#     def dev_edges(self, devnodes):
#         # node connectivity for dev edges
#         edge_conn_list = [(1, 2), (2, 3), (3, 4), (4, 1)]
#
#         # loop through device edges
#         nedge = 4
#         # initialize for loop
#         edges = []
#
#         for i in range(nedge):
#             # vertices node
#             i_start, i_end = edge_conn_list[i]
#             xdev0 = devnodes[i_start - 1, :]
#             xdevf = devnodes[i_end - 1, :]
#
#             edges.append(torch.tensor([xdev0, xdevf]))
#
#         return edges
#
#     def inpolygon(self, xq, yq, xv, yv):
#         num_queries = xq.size(0)
#         num_vertices = xv.size(0)
#
#         # Expand dimensions to enable broadcasting
#         xq = xq.unsqueeze(1)
#         yq = yq.unsqueeze(1)
#
#         # Create shifted versions of xv and yv
#         ## TODO: mps issue
#         xv_shifted = torch.roll(xv, shifts=1)
#         yv_shifted = torch.roll(yv, shifts=1)
#         # xv_shifted = torch.roll(xv.cpu(), shifts=1)
#         # yv_shifted = torch.roll(yv.cpu(), shifts=1)
#         # xv = xv
#         # yv = yv
#
#         # Compute winding numbers
#         winding_numbers = ((yv_shifted > yq) != (yv > yq)) & (
#                     xq < (xv - xv_shifted) * (yq - yv_shifted) / (yv - yv_shifted) + xv_shifted)
#
#         # Check if winding number is odd for each query point
#         in_results = winding_numbers.sum(dim=1) % 2 != 0
#
#         # Compute on-edge results
#         on_results = (yq == yv).any(dim=1) & ((xq >= xv).all(dim=1) | (xq <= xv_shifted).all(dim=1))
#
#         return in_results, on_results
#
#     def projectComp(self, mesh_centers, edges, devnodes):
#         # find elements within component bounding box
#         # project edges first
#         # center = p.domain * obj.centerCoord
#         # 3 devices, 4 edges/device
#         nedge = 4
#         # nedge = 4
#         nelx = self.nelx
#         nely = self.nely
#         nEl = self.nEl
#         rhoEdge = torch.zeros((nEl, nedge))
#
#         for i in range(nedge):
#             # edge coords
#             nodes = edges[i]
#             # edge for iter
#             x0, xf = nodes
#             w = 2 * self.rad
#             xfv = xf.repeat(mesh_centers.size(0), 1)
#             x0v = x0.repeat(mesh_centers.size(0), 1)
#
#             # preprocess
#             a = xfv - x0v
#             b = mesh_centers - x0v
#             e = mesh_centers - xfv
#
#             # nan issues
#             a = torch.where(torch.isnan(a), torch.zeros_like(a), a)
#             b = torch.where(torch.isnan(b), torch.zeros_like(b), b)
#             e = torch.where(torch.isnan(e), torch.zeros_like(e), e)
#
#             ghat = torch.stack([-a[:, 1], a[:, 0]], dim=1)
#             g = (torch.sum(ghat * b, dim=1) / torch.norm(ghat, dim=1, p=2).pow(2)).unsqueeze(1) * ghat
#             # g[torch.isnan(g)] = 0  # if x0&xf overlap then g = 0
#             g = torch.where(torch.isnan(g), torch.zeros_like(g), g)
#
#             aNorm = torch.norm(a, dim=1)
#             bNorm = torch.norm(b, dim=1)
#             eNorm = torch.norm(e, dim=1)
#             gNorm = torch.norm(g, dim=1)
#             # Find min distance
#             idxP1 = torch.nonzero(torch.sum(a * b, dim=1) <= 0).squeeze()
#             idxP2 = torch.nonzero(
#                 (torch.sum(a * b, dim=1) > 0) * (torch.sum(a * b, dim=1) < torch.sum(a * a, dim=1)) * (
#                             torch.sum(a * a, dim=1) != 0)).squeeze()
#             idxP3 = torch.nonzero(
#                 (torch.sum(a * b, dim=1) >= torch.sum(a * a, dim=1)) * (torch.sum(a * a, dim=1) != 0)).squeeze()
#
#             d = torch.zeros(self.nEl)
#             d[idxP1] = bNorm[idxP1]
#             d[idxP2] = gNorm[idxP2]
#             d[idxP3] = eNorm[idxP3]
#             # Calculate partial boundaries
#             phi_q = d - w / 2
#             idxR1 = torch.nonzero(phi_q > self.rad).squeeze()  # phi > r
#             idxR2 = torch.nonzero(torch.abs(phi_q) <= self.rad).squeeze()  # -r <= phi <= r
#             idxR3 = torch.nonzero(phi_q < -self.rad).squeeze()  # phi < -r
#             rho_q = torch.zeros(self.nEl)
#             # rho_q[idxR1] = torch.zeros_like(idxR1).float()
#             # rho_q[idxR1] = torch.zeros_like(idxR1)
#             # rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (self.rad ** 2 * torch.acos(phi_q[idxR2] / self.rad) - phi_q[idxR2] * torch.sqrt(self.rad ** 2 - phi_q[idxR2] ** 2))
#
#             # sqrt_tol = 1e-6
#             # tol_tens = torch.full_like(phi_q[idxR2], sqrt_tol)
#             # # Ensure that the input to torch.acos remains within [-1, 1]
#             # input_acos = torch.clamp(phi_q[idxR2] / self.rad, min=-1, max=1)
#             # # rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (self.rad ** 2 * torch.acos(phi_q[idxR2] / self.rad) - phi_q[idxR2] * torch.sqrt(torch.maximum((self.rad ** 2 - phi_q[idxR2] ** 2), tol_tens)))
#             # rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (self.rad ** 2 * torch.acos(input_acos) - phi_q[idxR2] * torch.sqrt(torch.maximum((self.rad ** 2 - phi_q[idxR2] ** 2), tol_tens)))
#             # prevent nan instabilities
#             input_acos = torch.clamp(phi_q[idxR2] / self.rad, min=-1, max=1)
#             input_sqrt = torch.clamp(self.rad ** 2 - phi_q[idxR2] ** 2, min=0.0)
#             rho_q[idxR2] = (1 / (math.pi * self.rad ** 2)) * (
#                         self.rad ** 2 * torch.acos(input_acos) - phi_q[idxR2] * torch.sqrt(input_sqrt))
#
#             # rho_q[idxR3] = torch.ones_like(idxR3).float()
#             rho_q[idxR3] = 1
#             # rho_q[idxR3] = torch.ones_like(idxR3)
#             # Convert to density
#             rho_tilde = self.rho_min + rho_q * (1 - self.rho_min)
#             rhoEdge[:, i] = rho_tilde
#
#         # rho_hat_proj = torch.linalg.norm(rhoEdge, ord=self.p, dim=1)
#         # rho_hat_proj = torch.norm(rhoEdge, self.p, dim=1)
#         # rho_hat_proj = torch.norm(rhoEdge, dim=1)
#         # rho_hat_proj = torch.norm(rhoEdge, dim=1)
#         # pytorch norm doesn't support p!=2 for some reason
#         # compute p norm manually
#         rho_hat_proj = torch.pow(torch.sum(torch.pow(torch.abs(rhoEdge), self.p), dim=1), 1.0 / self.p)
#
#         # print(rhoEdge.size())
#         # rho_hat_proj = torch.norm(rhoEdge, dim=1)
#
#         # # fill in inside of object
#         inpoly, onpoly = self.inpolygon(mesh_centers[:, 0], mesh_centers[:, 1], devnodes[:, 0], devnodes[:, 1])
#         # inIdx = torch.nonzero(inpoly.logical_and(~onpoly)).squeeze()
#         inIdx = torch.nonzero(inpoly.logical_and(~onpoly)).squeeze()
#
#         # rho_hat[inIdx] = torch.ones_like(rho_hat[inIdx])
#         # rho_hat_ones = torch.ones_like(rho_hat_proj[inIdx])
#         # rho_hat_proj[inIdx] = rho_hat_ones
#         # print(f"rho_hat_proj[inIdx].size(): {rho_hat_proj[inIdx].size()}")
#         # rho_hat_ones = torch.ones_like(rho_hat_proj[inIdx])
#         rho_hat_proj[inIdx] = 1
#
#         return rho_hat_proj
#
#     def rho_dev(self):
#
#         # initialize rho
#         rho_d = torch.zeros(self.nEl, self.ndev)
#
#         # project devices
#         ndev = 3
#         for j in range(ndev):
#             rho_d[:, j] = self.projectComp(self.centers, self.edges[j], self.dev_nodes[j])
#
#         # ############################
#         # # plot check
#         # # note, need permute since view() fills rows,
#         # # rather than columns in matlabs reshape()
#         # plt.imshow(rho.view(50, 50).permute(1,0).detach().numpy(), cmap='jet')
#         # # origin="lower"
#         # plt.axis('off')  # Turn off the axis labels
#         # plt.show()
#
#         # asdfasdf
#         # ############################
#
#         return rho_d
#
#     def rho_all(self):
#
#         # initialize rho
#         # rho = torch.zeros(self.nEl)
#         rho_hat = torch.zeros(self.nEl, self.nbars + self.ndev)
#
#         nbar = len(self.bar_nodes)
#
#         # project bars
#         for j in range(nbar):
#             nodes_j = self.bar_nodes[j]
#             width_j = self.widths[j]
#
#             rho_hat[:, j] = self.projectBar(nodes_j, width_j, self.centers)
#
#         # project devices
#         ndev = 3
#         for j in range(ndev):
#             rho_hat[:, nbar + j] = self.projectComp(self.centers, self.edges[j], self.dev_nodes[j])
#
#         # Merge overlapping members!
#         rho = torch.norm(rho_hat, self.p, dim=1)  # row-wise p norm
#
#         # # ############################
#         # # plot check
#         # # note, need permute since view() fills rows,
#         # # rather than columns in matlabs reshape()
#         # plt.imshow(rho.view(self.nelx, self.nely).permute(1,0).detach().cpu().numpy(), cmap='jet')
#         # # origin="lower"
#         # plt.axis('off')  # Turn off the axis labels
#         # plt.show()
#
#         # # asdfasdf
#         # # ############################
#
#         return rho