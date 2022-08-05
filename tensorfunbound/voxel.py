import torch


def custom_access(idx, t: torch.Tensor):
    D = t.dim()
    Ns = t.shape
    device = t.device
    idx = idx.to(device).float()
    for i in range(D):
        idx[:, i] = torch.clamp(idx[:, i], 0, Ns[i] - 1)
    idx = idx.long()
    return t[list(idx.T)]


class ModelVoxel(torch.nn.Module):
    def __init__(self, f=None, Ns=[128, 128], Ls=2.0, device="cpu", remap_needed=True):
        super().__init__()

        self.remap_needed = remap_needed

        if f is None:
            self.Ns = torch.tensor(Ns).to(device)
            self.f = torch.nn.Parameter(
                torch.randn(Ns, requires_grad=True, device=device)
            )
        else:
            self.f = torch.nn.Parameter(f.to(device))
            self.Ns = torch.tensor(f.shape).to(device)

        self.Ls = torch.tensor(Ls).to(device)
        self.hs = self.Ls / (self.Ns - 1)

    def remap(self, pts):
        device = pts.device

        return (pts + torch.Tensor([1.0, 1, 1]).to(device)) / self.hs[
            0
        ]  # NOT UNIVERSAL, FIX ME

    def forward(self, x):
        if self.remap_needed:
            x = self.remap(x)

        sh = list(x.shape)
        idx = x.reshape(-1, sh[-1])
        ans = custom_access(idx=idx, t=self.f)

        ans = ans.reshape(*sh[:-1])
        return ans


def DF_DX(ss: torch.Tensor, hx, dim=0):
    """ss: d-dimensional array  """

    DX = (
        torch.roll(ss, shifts=[1], dims=[dim]) - torch.roll(ss, shifts=[-1], dims=[dim])
    ) / (2 * hx)

    selectionA0 = [slice(None)] * 3
    selectionA1 = [slice(None)] * 3
    selectionA2 = [slice(None)] * 3

    selectionA0[dim] = 0
    selectionA1[dim] = 1
    selectionA2[dim] = 2

    DX[selectionA0] = (
        -(3 / 2) * ss[selectionA0] + 2 * ss[selectionA1] - 1 / 2 * ss[selectionA2]
    ) / hx

    selectionB0 = [slice(None)] * 3
    selectionB1 = [slice(None)] * 3
    selectionB2 = [slice(None)] * 3

    selectionB0[dim] = -1
    selectionB1[dim] = -2
    selectionB2[dim] = -3

    DX[selectionB0] = (
        -(3 / 2) * ss[selectionB0] + 2 * ss[selectionB1] - 1 / 2 * ss[selectionB2]
    ) / hx
    return -DX
