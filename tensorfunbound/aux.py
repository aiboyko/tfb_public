import time
import torch
import tntorch as tn
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ssp
from scipy import ndimage
import copy


def vis(d, levels=None):
    """mainly for plotting eikonal-type scalar fields"""
    A = np.max(d)
    if levels is None:
        levels = np.linspace(0, A, 40)
    plt.figure()
    plt.imshow(d, cmap="viridis_r")
    plt.colorbar()
    plt.contour(d, colors=["white"], levels=levels)


def Dx_1ord(Nx=128, h=1):
    Dx = ssp.lil_matrix(ssp.diags([-np.ones(Nx), np.ones(Nx - 1)], [0, 1]))
    Dx[0, 0] = -1
    Dx[0, 1] = 1
    Dx[-1, -1] = 1
    Dx[-1, -2] = -1
    return Dx / h


def Dx_2ord(Nx=128, h=1):
    Dx = ssp.lil_matrix(ssp.diags([-np.ones(Nx - 1), np.ones(Nx - 1)], [-1, 1]) / 2)
    Dx[0, 0] = -3 / 2
    Dx[0, 1] = 2
    Dx[0, 2] = -1 / 2

    Dx[-1, -1] = -1 / 2
    Dx[-1, -2] = 2
    Dx[-1, -3] = -3 / 2
    return Dx / h


class PartialDerivArray2D(object):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self._create_partial_derivs()

    def _create_partial_derivs(self):
        Nx, Ny = self.shape
        self.DX_operator = ssp.kron(Dx_2ord(Nx), ssp.eye(Ny))
        self.DY_operator = ssp.kron(ssp.eye(Nx), Dx_2ord(Ny))

    def DX(self, u: np.array):
        return self.DX_operator.dot(u.flatten()).reshape(u.shape)

    def DY(self, u: np.array):
        return self.DY_operator.dot(u.flatten()).reshape(u.shape)


class PartialDerivArray(object):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self._create_partial_derivs()

    def _create_partial_derivs(self):
        Nx, Ny, Nz = self.shape
        DX_operator = ssp.kron(Dx_2ord(Nx), ssp.eye(Ny))
        self.DX_operator = ssp.kron(DX_operator, ssp.eye(Nz))
        DY_operator = ssp.kron(ssp.eye(Nx), Dx_2ord(Ny))
        self.DY_operator = ssp.kron(DY_operator, ssp.eye(Nz))
        DZ_operator = ssp.kron(ssp.eye(Nx), ssp.eye(Ny))
        self.DZ_operator = ssp.kron(DZ_operator, Dx_2ord(Nz))

    def DX(self, u: np.array):
        return self.DX_operator.dot(u.flatten()).reshape(u.shape)

    def DY(self, u: np.array):
        return self.DY_operator.dot(u.flatten()).reshape(u.shape)

    def DZ(self, u: np.array):
        return self.DZ_operator.dot(u.flatten()).reshape(u.shape)


def rotate_3d(u: np.array, angles=[0, 0, 0], mode="nearest", cval=0):
    """rotation is applied as R(0x) R(0y) R(0z)"""
    u = copy.copy(u)
    Nx, Ny, Nz = u.shape
    az, ay, ax = angles

    u = ndimage.rotate(u, az, axes=(0, 1), reshape=False, mode=mode)
    u = ndimage.rotate(u, ay, axes=(0, 2), reshape=False, mode=mode)
    u = ndimage.rotate(u, ax, axes=(1, 2), reshape=False, mode=mode)
    # if az!= 0:
    #     for iz in range(Nz):
    #         u[:,:,iz] = ndimage.rotate(u[:,:,iz], az)
    # if ay!= 0:
    #     for iy in range(Ny):
    #         u[:,iy,:] = ndimage.rotate(u[:,iy,:], ay)
    # if ax!= 0:
    #     for ix in range(Nx):
    #         u[ix,:,:] = ndimage.rotate(u[ix,:,:], ax)
    return u


def shift_3d(u: np.array, shift=[0, 0, 0], mode="nearest", cval=0):
    u = copy.copy(u)
    Nx, Ny, Nz = u.shape
    sx, sy, sz = shift
    if sx != 0 or sy != 0:
        for iz in range(Nz):
            u[:, :, iz] = ndimage.shift(
                u[:, :, iz], shift=[sx, sy], mode=mode, cval=cval
            )
    if sz != 0:
        for iy in range(Ny):
            u[:, iy, :] = ndimage.shift(
                u[:, iy, :], shift=[0, sz], mode=mode, cval=cval
            )
    return u


def gradient(t: tn.Tensor, h=1.0):
    "this is to replace gradient() in tntorch. It has higher order differentiatin AND no changing size of the grid"
    device = t.cores[0].device
    partials = []
    for dim in range(t.dim()):
        dx = Dx_2ord(Nx=t.shape[dim], h=h)
        dx = torch.tensor(dx.toarray(), dtype=torch.float32).to(device)
        c = torch.einsum("ij, ajk-> aik", dx, t.cores[dim])
        dxcores = copy.copy(t.cores)
        dxcores[dim] = c
        partials.append(tn.Tensor(dxcores))
    return partials


def device_check(tt: tn.Tensor):
    print(tt.cores[0].device)


def printmaxrank(tt: tn.tensor, eps:float):
    maxrank = max(tt.ranks_tt)
    print('maxTTrank =', maxrank, '(with eps = ', eps, ')')