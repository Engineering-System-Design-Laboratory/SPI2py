import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import math



# Mesh parameters
L_x
L_y
nelx
nely

def kron(self, a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = (a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4))
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

def Reg_Mesh(self, L_x, L_y, nelx, nely):
    # def Reg_Mesh(self):
    # Define number of elements and nodes
    # L_x = self.L_x
    # L_y = self.L_y
    # nelx = self.nelx
    # nely = self.nely
    numEL = nelx * nely
    numNODE = (nelx + 1) * (nely + 1)
    dx = L_x / nelx
    dy = L_y / nely

    # Determine nodal coordinates
    X, Y = torch.meshgrid(torch.linspace(0, L_x, nelx + 1), torch.linspace(L_y, 0, nely + 1))
    coords = torch.column_stack((X.reshape((nelx + 1) * (nely + 1), 1), Y.reshape((nelx + 1) * (nely + 1), 1))).to(
        device)

    # Determine element degrees of freedom
    nodenrs = (torch.arange((1 + nelx) * (1 + nely)) + 1).reshape(1 + nely, 1 + nelx)
    edofVec = torch.reshape((nodenrs[0:-1, 0:-1] + 1), (nelx * nely, 1))
    edofMat = torch.tile(edofVec, (1, 4)) + torch.tile(torch.tensor([0, nely + 1, nely, -1]), (nelx * nely, 1)).to(
        device)

    # Determine DOF mapping
    # TODO: double check flatten works too
    # iK = torch.reshape(torch.kron(edofMat, torch.ones((4, 1))), (16 * nelx * nely,))
    # jK = torch.reshape(torch.kron(edofMat, torch.ones((1, 4))), (16 * nelx * nely,))
    # iP = torch.reshape(edofMat.T, (4 * nelx * nely,))
    # iP = edofMat.view(1, -1)
    iK = torch.kron(edofMat, torch.ones((4, 1))).flatten()
    jK = torch.kron(edofMat, torch.ones((1, 4))).flatten()
    iP = edofMat.flatten()

    # print(f"edofMat: {edofMat}")
    # print(f"edofMat.view(1, -1): {edofMat.view(1, -1)}")
    # print(f"iP: {iP}")
    # asdfasdf

    # Determine element centers
    centers = torch.zeros((numEL, 2))
    centers[:, 0] = (coords[edofMat[:, 1] - 1, 0] + coords[edofMat[:, 0] - 1, 0]) / 2
    centers[:, 1] = (coords[edofMat[:, 3] - 1, 1] + coords[edofMat[:, 0] - 1, 1]) / 2

    return coords, edofVec, edofMat, iK, jK, iP, centers

def forward(self):
    coords, edofVec, edofMat, iK, jK, iP, centers = self.Reg_Mesh(self.L_x, self.L_y, self.nelx, self.nely)
    # coords, edofVec, edofMat, iK, jK, iP, centers = self.Reg_Mesh()
    return coords, edofVec, edofMat, iK, jK, iP, centers


# FEA Parameters
L_x = 1
L_y = 1
nelx = 1
nely = 1
ndof = (nelx + 1) * (nely + 1)
numEL = nelx * nely
coords = 1
edofVec = 1
edofMat = 1
iK = 1
jK = 1
iP = 1
xProj = 1
dim = 2
kappa = 54  # thermal conductivity of steel @ .5C (when rho = 1)
Q = 1  # internal heat generation
kapMin = .03162  # thermal conductivity of air @ 100C (at min rho)
densMin = kapMin / kappa
penal = 3
nComp = 3
rho_d = rho_d = 1
dev_Q = 1
pIdx = 1
dev_cent = 1
mesh_centers = 1
Tp = 100  # prescribed temperature

def FEA_shapeFcn(self, i, gPoints):
    ipt_coords = torch.tensor([gPoints[i, 0], gPoints[i, 1]])
    N = 1 / 4 * torch.tensor([(1 - ipt_coords[0]) * (1 - ipt_coords[1]),
                              (1 + ipt_coords[0]) * (1 - ipt_coords[1]),
                              (1 + ipt_coords[0]) * (1 + ipt_coords[1]),
                              (1 - ipt_coords[0]) * (1 + ipt_coords[1])])

    dN = 1 / 4 * torch.tensor([[-(1 - ipt_coords[1]), -(1 - ipt_coords[0])],
                               [(1 - ipt_coords[1]), -(1 + ipt_coords[0])],
                               [(1 + ipt_coords[1]), (1 + ipt_coords[0])],
                               [-(1 + ipt_coords[1]), (1 - ipt_coords[0])]])

    return N, dN

def element_comp(self, coords, edofMat):
    sqrt_3 = torch.sqrt(torch.tensor(3))
    gPoints = torch.tensor([[-1 / sqrt_3, -1 / sqrt_3, -1 / sqrt_3],
                            [1 / sqrt_3, -1 / sqrt_3, -1 / sqrt_3],
                            [1 / sqrt_3, 1 / sqrt_3, -1 / sqrt_3],
                            [-1 / sqrt_3, 1 / sqrt_3, -1 / sqrt_3],
                            [-1 / sqrt_3, -1 / sqrt_3, 1 / sqrt_3],
                            [1 / sqrt_3, -1 / sqrt_3, 1 / sqrt_3],
                            [1 / sqrt_3, 1 / sqrt_3, 1 / sqrt_3],
                            [-1 / sqrt_3, 1 / sqrt_3, 1 / sqrt_3]])

    wt = torch.tensor([1, 1, 1, 1])

    elem_coords = torch.stack((coords[edofMat[0, :] - 1, 0],
                               coords[edofMat[0, :] - 1, 1]), dim=0)

    # Physics definitions
    D = torch.diag(torch.ones(self.dim) * self.kappa)
    k_el = torch.zeros(2 ** self.dim, 2 ** self.dim)
    p_el = torch.zeros(2 ** self.dim)

    for i in range(2 ** self.dim):
        N, dN = self.FEA_shapeFcn(i, gPoints)

        J = torch.matmul(elem_coords, dN)
        ## TODO: mps issue
        detJ = torch.det(J)
        # detJ = torch.det(J.cpu())
        B = torch.matmul(dN, torch.inverse(J))
        k_el += torch.matmul(torch.matmul(B, D), B.transpose(0, 1)) * detJ * wt[i]
        p_el += self.Q * N * detJ * wt[i]

    return k_el, p_el

# need to call inputs here
def dev_load(self):

    K_el, P_el = self.element_comp(self.coords, self.edofMat)
    # Assemble conductive K
    sK = (K_el.view(1, -1) * (self.densMin + (1 - self.densMin) * self.xProj.view(-1, 1) ** self.penal)).flatten()
    indices = torch.vstack((self.iK, self.jK)) - 1

    # # allocate some memory on the GPU
    # x = torch.randn(50000, 50000)

    # # get the maximum GPU memory allocated by PyTorch
    # max_memory_allocated = torch.cuda.max_memory_allocated()

    # # print the result
    # print(f"Maximum GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")

    K = torch.sparse_coo_tensor(indices, sK, size=[self.ndof, self.ndof],
                                dtype=torch.double).to_dense()
    # TODO: mps issue

    # K = torch.sparse_coo_tensor(indices.cpu(), sK.cpu(), size=[self.ndof, self.ndof]).to_dense()

    # asdfasdfasdf
    # TODO
    # K = torch.sparse_coo_tensor(indices, sK, size=[self.ndof, self.ndof], dtype=torch.double)
    # temptest = torch.zeros([self.ndof, self.ndof])
    # asdfasdf
    K = (K + K.t()) / 2

    ########################
    # Assemble load vector

    # P = torch.zeros(self.ndof, 1)
    # print(f"self.ndof: {self.ndof}")
    # print(f"P.size(): {P.size()}")

    indices = torch.stack([self.iP - 1, torch.zeros_like(self.iP, dtype=torch.long)])
    P_log = []

    for i in range(1, self.nComp + 1):
        Q = self.dev_Q[i - 1]
        sP = (Q * P_el * (self.rho_d[:, i - 1].view(-1, 1) ** self.penal)).view(-1,
                                                                                                      1).flatten()
        # P = P + torch.sparse_coo_tensor(indices, sP, size=P.size())
        # P += torch.sparse_coo_tensor(indices, sP, size=torch.zeros(self.ndof, 1).size()).to_dense()
        P_iter = torch.sparse_coo_tensor(indices, sP, size=torch.zeros(self.ndof, 1).size()).to_dense()
        P_log.append(P_iter)

    P_stack = torch.stack(P_log, dim=0)
    P = torch.sum(P_stack, dim=0)

    ########################

    # ## TODO: mps issue

    # P = torch.zeros(self.ndof, 1)
    # # Plog = []

    # indices = torch.stack([self.iP - 1, torch.zeros_like(self.iP, dtype=torch.long)]).cpu()

    # for i in range(1, self.nComp + 1):
    #     Q = self.dev_Q[i - 1]
    #     sP = (Q * P_el * (self.rho_d[:, i - 1].view(-1, 1)**self.penal)).view(-1,1).flatten().cpu()
    #     maxval, maxidx = torch.max(sP, dim=0)
    #     # P = P + torch.sparse_coo_tensor(indices, sP, size=P.size())
    #     ## TODO: mps issue
    #     # print(f"P: {P}")
    #     # print(f"indices: {indices}")
    #     # print(f"sP: {sP}")
    #     # P = P + torch.sparse_coo_tensor(indices, sP, size=P.size())
    #     P_it = torch.sparse_coo_tensor(indices, sP, size=self.ndof)
    #     print(f"P_it: {P_it}")
    #     P += P_it



    # square_P = P.to_dense().reshape(self.nelx + 1, self.nely + 1).T.cpu()
    # square_P = P.to_dense().reshape(101, 101).T.cpu()

    # return P.to_dense(), K.float()
    return P, K
    # # # TODO: mps issue
    # # return P.to_dense(), K.float()
    # return P, K

def forward(self):
    P, K = self.dev_load()

    # Set FEA.fixedDOFS to pIdx
    fixedDOFs = self.pIdx

    # Calculate FEA.Up
    Up = self.Tp * torch.ones_like(self.pIdx, dtype=torch.float32)
    # Up = self.Tp * torch.ones_like(self.pIdx)

    # Calculate freeDOFS using setdiff (complement of pIdx)
    # all_nodes = torch.arange(1, self.ndof + 1)
    all_nodes = torch.arange(self.ndof)
    combined = torch.cat((all_nodes, fixedDOFs))
    uniquedof, countdof = combined.unique(return_counts=True)
    freeDOFs = uniquedof[countdof == 1]
    intersectdof = uniquedof[countdof > 1]

    # solve
    U = torch.zeros(self.ndof)

    # Assign FEA.Up to the corresponding indices in U
    U[fixedDOFs] = Up

    # square_U = U.reshape(self.nelx + 1, self.nely + 1).T.cpu()
    # square_U = U.reshape(101, 101).T.cpu()

    # Calculate U for FEA.freeDOFS
    # asdfasdf
    # U[freeDOFs] = torch.solve(P[freeDOFs] - torch.matmul(K[freeDOFs, fixedDOFs], Up), K[freeDOFs, freeDOFs]).solution

    # TODO: check this
    # Ksolve = K[freeDOFs, freeDOFs]
    Ksolve = K[freeDOFs[:, None], freeDOFs].float()
    # # Psolve = (P[freeDOFs].squeeze() - torch.matmul(K[freeDOFs.unsqueeze(1), fixedDOFs], Up.double())).unsqueeze(1)

    Psolve = (P[freeDOFs].squeeze() - torch.matmul(K[freeDOFs.unsqueeze(1), fixedDOFs].float(), Up)).unsqueeze(1)

    # # U[freeDOFs] = torch.linalg.solve(K[freeDOFs, freeDOFs], (P[freeDOFs].squeeze() - torch.matmul(K[freeDOFs.unsqueeze(1), fixedDOFs], Up.double())).unsqueeze(1))
    # # U[freeDOFs] = torch.linalg.solve(Ksolve, Psolve)

    Usolve = torch.linalg.solve(Ksolve, Psolve)
    # ## TODO: mps issue
    # Usolve = torch.linalg.solve(Ksolve.cpu(), Psolve.cpu())
    # print(f"Usolve.size(): {Usolve.size()}")

    U[freeDOFs] = Usolve.squeeze()


    # square_U = U.reshape(self.nelx + 1, self.nely + 1).T.cpu()

    # # Create a finer mesh using interpolation
    # x = np.linspace(0, self.nelx, (self.nelx + 1) * 10)  # Create a finer x-axis
    # y = np.linspace(0, self.nely, (self.nely + 1) * 10)  # Create a finer y-axis
    # X, Y = np.meshgrid(x, y)  # Create a meshgrid

    # # Interpolate the data using a smoother method, e.g., cubic interpolation
    # f = interp2d(np.arange(self.nelx + 1), np.arange(self.nely + 1), square_U, kind='linear')
    # Z = f(x, y)  # Evaluate the smoothed data on the finer mesh

    # asdfasdf
    # # Update P for FEA.fixedDOFS
    # P[fixedDOFs] = torch.matmul(K[fixedDOFs, fixedDOFs], U[fixedDOFs]) + torch.matmul(K[fixedDOFs, freeDOFs], U[freeDOFs])
    dev_temp = []
    # compute constraint here
    for i in range(self.nComp):
        # conIdx = 1 + p.g.nLength + p.g.nInt + p.g.ndvcDvc + p.g.ndvcBar + p.g.nElRad + i
        # device center

        dev_center = self.dev_cent[i]
        # find closest mesh center
        cv = dev_center.repeat(self.mesh_centers.size(0), 1)
        dists = torch.norm(self.mesh_centers - cv, dim=1)
        el = torch.argmin(dists)
        dof = self.edofMat[el] - 1
        # calculate temp at device center
        # double check that
        Ue = U[dof]

        elem_coords = self.coords[self.edofMat[0]]
        pt = dev_center - self.mesh_centers[el]
        dx = 1 / self.nelx
        dy = 1 / self.nely
        pt[0] = (2 / dx) * pt[0]
        pt[1] = (2 / dy) * pt[1]
        N, DN = self.FEA_shapeFcn(0, pt.unsqueeze(0))
        # J = torch.matmul(elem_coords.T, DN)

        # # B = DN / J
        # B = DN @ torch.inverse(J)
        T = torch.matmul(N, Ue)
        dev_temp.append(T)

    return dev_temp




