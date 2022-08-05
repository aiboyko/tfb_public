import numpy as np
import torch
import tntorch as tn
import copy


def custom_access_float(idx: torch.Tensor, t: tn.Tensor, interpolation="closest"):
    """idx is a 2D np.array of size Nbatch * D"""
    # print(idx)
    # print('--------')
    # print(idx.shape)
    def ttcontract(cores: list):
        while len(cores) > 2:
            cores[-2] = torch.einsum("abc,cbk->abk", cores[-2], cores[-1])
            del cores[-1]
        return torch.einsum("abc,cbk->b", cores[0], cores[1])

    D = t.dim()
    Ns = t.shape
    device = t.cores[0].device
    # print(idx)
    idx = idx.to(device).float()
    # print(idx, t.shape, idx.shape)
    if interpolation == "closest":
        for i in range(D):
            idx[:, i] = torch.clamp(idx[:, i], 0, Ns[i] - 1)
        idx = idx.int()
        return t[
            idx
        ].torch()  # can use .cores[0].view(-1) instead of .torch(); works 1.5x faster but not fully tested

    elif interpolation == "linear":
        D = t.dim()
        for i in range(D):
            idx[:, i] = torch.clamp(idx[:, i], 0, Ns[i] - 1)
        idx_low = torch.floor(idx).long()
        idx_high = torch.ceil(idx).long()

        wh = idx - idx_low

        newcores = []
        for i in range(D):
            newcores.append(
                (1 - wh[:, i]).reshape(-1, 1) * t.cores[i][:, idx_low[:, i], :]
                + (wh[:, i]).reshape(-1, 1) * t.cores[i][:, idx_high[:, i], :]
            )
        return ttcontract(newcores)

    else:
        raise ValueError


class ModelConTT(torch.nn.Module):
    def __init__(
        self,
        tt: tn.Tensor = None,
        interpolation="closest",
        RMAX=16,
        Ns=[128, 128],
        Ls=None,
        device="cpu",
        remap_needed=True,
    ):

        super().__init__()
        self.Ns = torch.tensor(Ns).to(device)
        self.Ls = Ls
        self.remap_needed = remap_needed

        if tt is None:
            self.tt = tn.randn(Ns, ranks_tt=RMAX, requires_grad=True, device=device)
            print("none")
        else:
            self.tt = tt
            self.Ns = torch.tensor(tt.shape).to(device)

        if self.Ls is not None:
            self.Ls = torch.tensor(Ls).to(device)

        self.interpolation = interpolation
        for i in range(self.tt.dim()):
            self.register_parameter(str(i), torch.nn.Parameter(self.tt.cores[i]))

        self.tt.cores = list(self.parameters())
        self = self.to(device)

    def remap(self, pts):
        device = pts.device
        Nx = self.Ns[0].to(device)

        return (
            (pts + torch.tensor([1.0], device=device)) / 2 * (Nx - 1)
        )  # NOT UNIVERSAL, FIX ME

    def forward(self, x):
        if self.remap_needed:
            x = self.remap(x)

        if self.Ls is not None:
            x = x / self.Ls * (self.Ns - 1)
        """x is BxD"""
        sh = list(x.shape)
        ans = custom_access_float(
            idx=x.reshape(-1, sh[-1]), t=self.tt, interpolation=self.interpolation
        )

        ans = ans.reshape(*sh[:-1])
        return ans


class DoubleModel(torch.nn.Module):
    #created mainly to store both SDF and TSDF in one model
    def __init__(
        self,
        tt_inner,
        tt_outer,
        mu,
        tt_mask=None,
        device="cpu",
        remap_needed=False,
        interpolation="closest",
        Ls_inner=None,
        Ls_outer=None
    ):
        super().__init__()
        self.device = device

        if Ls_inner is None and Ls_outer is None:
            self.model_inner = ModelConTT(
                tt=tt_inner,
                device=device,
                interpolation=interpolation,
                remap_needed=remap_needed,
            )
            self.model_outer = ModelConTT(
                tt=tt_outer,
                device=device,
                interpolation=interpolation,
                remap_needed=remap_needed,
            )
        else:
            self.model_inner = ModelConTT(
                tt=tt_inner,
                device=device,
                interpolation=interpolation,
                remap_needed=remap_needed,
                Ls=Ls_inner
            )
            self.model_outer = ModelConTT(
                tt=tt_outer,
                device=device,
                interpolation=interpolation,
                remap_needed=remap_needed,
                Ls=Ls_outer
            )

        self.mu = mu
        # self.remap_needed = remap_needed
        if tt_mask is None:
            self.model_mask = self.model_outer
        else:
            self.model_mask = ModelConTT(
                tt=tt_mask,
                device=device,
                interpolation=interpolation,
                remap_needed=remap_needed,
            )

    # def remap(self, pts):
    #     device = pts.device
    #     Nx = self.model_inner.Ns[0]
    #     return (
    #         (pts + torch.Tensor([1.0, 1, 1]).to(device)) / 2 * (Nx-1)
    #     )  # NOT UNIVERSAL, FIX ME

    def forward(self, pts):
        # if self.remap_needed:
        #     pts = self.remap(pts)
        vals_inner = self.model_inner(pts)
        vals_outer = self.model_outer(pts)
        D = self.model_mask(pts)
        mask = D.abs() < self.mu

        return vals_inner * mask + vals_outer * ~mask


def grad(pts, models_grads: list):
    """A helper function needed in case we store partial
     derivatives as a list of 3 models"""
    return torch.vstack(
        [models_grads[0](pts), models_grads[1](pts), models_grads[2](pts)]
    ).T
