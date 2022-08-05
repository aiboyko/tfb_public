import torch
from tensorfunbound.main import *

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