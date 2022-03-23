import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvGRU(nn.Module):
    def __init__(self, input_dim=3, nc_out=3, hidden_dim=64,
                kernel_size=3, batch_first=True, n_cell=3, dtype=torch.cuda.FloatTensor):
        super(ConvGRU, self).__init__()
        pad = kernel_size // 2
        self.dtype = dtype
        self.n_cell = n_cell
        self.batch_first = batch_first

        self.encode0 = ResidualBlock(input_dim)
        self.encode1a = Encode(input_dim, hidden_dim, kernel_size=3, downsample=2)
        self.encode1b = Encode(input_dim, hidden_dim, kernel_size=3, downsample=2)
        self.encode2a = ResidualBlock(hidden_dim)
        self.encode2b = ResidualBlock(hidden_dim)
        self.encode3 = ResidualBlock(2*hidden_dim)
        self.encode4 = Encode(2*hidden_dim, 4*hidden_dim, kernel_size=3, downsample=2)

        self.decode1a = Decode(4*hidden_dim, 2*hidden_dim, kernel_size=3)
        self.decode1b = Decode(4*hidden_dim, 2*hidden_dim, kernel_size=3, upsample=2)
        self.decode2a = Decode(2*hidden_dim, hidden_dim, kernel_size=3, upsample=2)
        self.decode2b = Decode(2*hidden_dim, hidden_dim, kernel_size=3)
        self.decode3a = Decode(hidden_dim, input_dim, kernel_size=3, upsample=2)
        self.decode3b = Decode(hidden_dim, input_dim, kernel_size=3, upsample=2)
        self.decode4 = Decode(2*input_dim, input_dim, kernel_size=3)
        self.res_block = ResidualBlock(input_dim)
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(input_dim, input_dim, kernel_size=3)
            )

        cell_list = []
        for i in range(self.n_cell):
            cell_list.append(GRU2D(input_dim=4*hidden_dim, hidden_dim=4*hidden_dim,
                                    kernel_size=5, bias=True))
        self.cell_list = nn.ModuleList(cell_list)
    
    def init_state(self, size):
        return Variable(torch.zeros(size).type(self.dtype))

    def forward(self, x, prev_state=None):
        if not self.batch_first:
            x = x.permute(1,0,2,3,4)
        x = x.squeeze()

        x = self.encode0(x)
        x1 = self.encode1a(x)
        x2 = self.encode1b(x)
        x1 = self.encode2a(x1)
        x2 = self.encode2b(x2)
        x = self.encode3(torch.cat([x1,x2],dim=1))
        x = self.encode4(x)
        x = x.unsqueeze(0)

        if prev_state is None:
            h = self.init_state(x[:,1,:,:,:].size())
        hidden = [h] * (x.size(1) + 1)
        for layer_idx in range(self.n_cell):
            for t in range(x.size(1)):
                hidden[t+1] = self.cell_list[layer_idx](x[:,t,:,:,:], hidden[t])
        hidden = hidden[-1]
        
        # h = h.squeeze()
        x1 = self.decode1a(hidden)
        x2 = self.decode1b(hidden)
        x1 = self.decode2a(x1)
        x2 = self.decode2b(x2)
        x1 = self.decode3a(x1)
        x2 = self.decode3b(x2)
        x = self.decode4(torch.cat([x1,x2],dim=1))
        x = self.res_block(x)
        out = self.out_conv(x)
        return out

class Encode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=None):
        super(Encode, self).__init__()
        if not downsample: downsample = 1
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(out_channels, out_channels, kernel_size, downsample),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

        
class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample=None):
        super(Decode, self).__init__()
        if not upsample: upsample = 1
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Encode(channels, channels, kernel_size=3)
        self.conv2 = Encode(channels, channels, kernel_size=3)
        self.relu = nn.PReLU()
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + x
        return out

class GRU2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, bias=True):
        super(GRU2D, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([x, reset_gate*h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h + update_gate * cnm
        return h_next

if __name__ == '__main__':
    from torchinfo import summary
    model = ConvGRU()
    summary(model, input_size=(1,5,3,72,128))