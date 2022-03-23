import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x) + x

class FB(nn.Module):
    def __init__(self, in_channels=3, mid1=20, mid2=40, out_channels=60):
        super().__init__()
        self.fb = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid1,
                        kernel_size=5, padding=2),
            nn.Conv2d(in_channels=mid1, out_channels=mid2,
                        kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=mid2, out_channels=out_channels,
                        kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        return self.fb(x)

class FR(nn.Module):
    def __init__(self, num_blocks=6, channels=80):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(channels))
        self.fr = nn.Sequential(*layers)

    def forward(self, x):
        return self.fr(x)

class FL(nn.Module):
    def __init__(self, channels=80):
        super().__init__()
        self.fl = nn.Sequential(
            nn.ConvTranspose2d(channels, channels//2,
                                kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels//2, channels//4,
                                kernel_size=4, stride=2, padding=1),
            nn.Conv2d(channels//4, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.fl(x)

class Fh(nn.Module):
    def __init__(self, channels=80):
        super().__init__()
        self.fh = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=3, padding=1),
            ResBlock(channels//4),
            nn.Conv2d(channels//4, channels//4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.fh(x)

class RNN(nn.Module):
    def __init__(self, num_cell=1, iteration=1, hidden_dim=80):
        super().__init__()
        # self.num_cell = num_cell
        # self.iteration = iteration
        
        # h_dim = hidden_dim // 4
        # latent_dim = hidden_dim - h_dim

        self.FB = FB()
        self.FR = FR()
        self.FL = FL()
        self.Fh = Fh()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x, h):
        fbt = self.FB(x)
        state = torch.cat([fbt, h], dim=1)
        fr = self.FR(state)
        L = self.FL(fr)
        h = self.Fh(fr)
        return L, h


if __name__ == '__main__':
    model = RNN()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))