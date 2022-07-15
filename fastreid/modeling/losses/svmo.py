import torch
import torch.nn as nn


class SVMORegularizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.beta = 1e-3

    def dominant_eigenvalue(self, A):

        N, _ = A.size()
        x = torch.rand(N, 1, device='cuda')

        Ax = (A @ x)
        AAx = (A @ Ax)

        return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)


    def get_singular_values(self, A):

        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)
        I = torch.eye(N, device='cuda')  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)
        return tmp + largest, largest

    def forward(self, W):

        # old_W = W
        old_size = W.size()

        if old_size[0] == 1:
            return 0

        W = W.view(old_size[0], -1).permute(1, 0)  # (C x H x W) x S

        smallest, largest = self.get_singular_values(W)
        return (
            self.beta * 10 * (largest - smallest)**2
        ).squeeze()