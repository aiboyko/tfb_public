import torch
import numpy as np

class ModelConTucker(torch.nn.Module):
    def __init__(self, shape, rank_max, device, interpolation="closest"):
        super().__init__()
        self.shape = shape

        U = torch.rand([rank_max] * len(shape), requires_grad=True, device=device)
        Qs = [
            torch.rand((i, rank_max), requires_grad=True, device=device) for i in shape
        ]

        self.U = torch.nn.Parameter(U)
        self.Qs = torch.nn.ParameterList([torch.nn.Parameter(Q) for Q in Qs])
        self.interpolation = interpolation
        # self.cores = torch.nn.ParameterDict({'U':U, 'Q': Qs})

    def contract(self, idx):
        res = self.U
        # print(self.Qs[-1][idx[:,-1]].shape)
        # print(res.shape)
        res = torch.einsum("...i,ji -> j...", res, self.Qs[-1][idx[:, -1]])
        # print(res.shape)
        for i, Q in enumerate(self.Qs[1:-1][::-1]):
            res = torch.einsum("j...i,ji -> j...", res, Q[idx[:, i]])
        res = torch.einsum("ji,ji -> j", res, self.Qs[0][idx[:, 0]])
        return res

    def contract_full(self):
        res = self.U
        # print(self.Qs[-1][idx[:,-1]].shape)
        # print(res.shape)
        # print(res.shape)
        for i, Q in enumerate(self.Qs[::-1]):
            res = torch.einsum("...i,ji -> j...", res, Q)
        return res

    def custom_access(self, idx, sh):
        idx = idx.float()
        D = sh[-1]
        if self.interpolation == "closest":
            for i in range(D):
                idx[:, i] = np.clip(idx[:, i], 0, self.shape[i] - 1)
            return self.contract(idx.long())
        else:
            raise ValueError

    def forward(self, x):
        sh = list(x.shape)
        res = self.custom_access(x, sh)
        return res


# class ModelConTucker(torch.nn.Module):
#     def __init__(self, shape, rank_max, device, interpolation='closest'):
#         super().__init__()
#         self.shape = shape

#         U = torch.rand([rank_max]*len(shape), requires_grad=True, device=device)
#         Qs = [torch.rand((i, rank_max), requires_grad=True,device=device) for i in shape[::-1]]

#         self.U = torch.nn.Parameter(U)
#         self.Qs = torch.nn.ParameterList([torch.nn.Parameter(Q) for Q in Qs])
#         self.interpolation = interpolation
#         #self.cores = torch.nn.ParameterDict({'U':U, 'Q': Qs})

#     def contract(self, idx):
#         res = self.U
#         #print(self.Qs[-1][idx[:,-1]].shape)
#         #print(res.shape)
#         res = torch.einsum("...i,ji -> j...", res, self.Qs[-1][idx[:,-1]])
#         #print(res.shape)
#         for i, Q in enumerate(self.Qs[1:-1]):
#             res = torch.einsum("j...i,ji -> j...", res, Q[idx[:,-2-i]])
#         res = torch.einsum("ji,ji -> j", res, self.Qs[0][idx[:,0]])
#         return res

#     def contract_full(self):
#         res = self.U
#         #print(self.Qs[-1][idx[:,-1]].shape)
#         #print(res.shape)
#         #print(res.shape)
#         for i, Q in enumerate(self.Qs):
#             res = torch.einsum("...i,ji -> j...", res, Q)
#         return res

#     def custom_access(self, idx, sh):
#         idx = idx.float()
#         D = sh[-1]
#         if self.interpolation == "closest":
#             for i in range(D):
#                 idx[:, i] = np.clip(idx[:, i], 0, self.shape[i] - 1)
#             return self.contract(idx.long())
#         else:
#             raise ValueError

#     def forward(self, x):
#         sh = list(x.shape)
#         res = self.custom_access(x, sh)
#         return res
