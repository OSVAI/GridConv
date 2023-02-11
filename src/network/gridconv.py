import torch.nn as nn

class GridLiftingNetwork(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_block=2,
                 num_jts=17,
                 out_num_jts=17,
                 p_dropout=0.25,
                 input_dim=2,
                 output_dim=3):
        super(GridLiftingNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_stage = num_block
        self.num_jts = num_jts
        self.out_num_jts = num_jts
        self.out_dim = output_dim
        self.p_dropout = p_dropout

        self.input_size = num_jts * input_dim
        self.output_size = out_num_jts * output_dim

        conv = TwoBranchGridConv

        self.w1 = conv(in_channels=2, out_channels=self.hidden_size, kernel_size=3, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(self.hidden_size)
        self.dropout = nn.Dropout2d(p=self.p_dropout)

        self.linear_stages = []
        for l in range(num_block):
            self.linear_stages.append(CNNBlock(self.hidden_size, p_dropout=self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = conv(in_channels=self.hidden_size, out_channels=3, kernel_size=3, bias=False)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y

class CNNBlock(nn.Module):
    def __init__(self, hidden_size, p_dropout=0.25, biased=True):
        super(CNNBlock, self).__init__()
        self.hid_size = hidden_size

        self.relu = nn.ReLU(inplace=True)

        conv = TwoBranchGridConv

        self.w1 = conv(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, bias=biased)
        self.batch_norm1 = nn.BatchNorm2d(self.hid_size)

        self.w2 = conv(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, bias=biased)
        self.batch_norm2 = nn.BatchNorm2d(self.hid_size)

        self.dropout = nn.Dropout2d(p=p_dropout)


    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class TwoBranchGridConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(TwoBranchGridConv, self).__init__()
        self.donut_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    padding=[kernel_size // 2 * 2, kernel_size // 2 * 2], padding_mode='circular', bias=bias)
        self.tablet_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=[kernel_size // 2, kernel_size // 2], padding_mode='zero', bias=bias)

    def forward(self, x):
        y_cir = self.donut_conv(x)
        y_rep = self.tablet_conv(x)
        out = y_cir + y_rep

        return out


