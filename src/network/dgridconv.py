import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.nn.init as init

class DynamicGridLiftingNetwork(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_block=2,
                 num_jts=17,
                 out_num_jts=17,
                 p_dropout=0.25,
                 input_dim=2,
                 output_dim=3,
                 grid_shape=(5,5),
                 temperature=30,
                 padding_mode=('c','r')):
        super(DynamicGridLiftingNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_stage = num_block
        self.num_jts = num_jts
        self.out_num_jts = num_jts
        self.out_dim = output_dim
        self.p_dropout = p_dropout

        self.input_size = num_jts * input_dim
        self.output_size = out_num_jts * output_dim

        conv = TwoBranchDGridConv

        self.w1 = conv(in_channels=2, out_channels=self.hidden_size, kernel_size=3, padding_mode=padding_mode, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(self.hidden_size)
        self.dropout = nn.Dropout2d(p=self.p_dropout)

        self.atten_conv1 = DynamicAttention2D(in_planes=input_dim, out_planes=self.hidden_size, grid_shape=grid_shape, kernel_size=3,
                                              ratios=1/16., temperature=temperature, groups=1)
        self.linear_stages = []
        for l in range(num_block):
            self.linear_stages.append(CNNBlock(self.hidden_size, grid_shape=grid_shape, padding_mode=padding_mode, p_dropout=self.p_dropout, temperature=temperature))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = conv(in_channels=self.hidden_size, out_channels=3, kernel_size=3, padding_mode=padding_mode, bias=False)

        self.atten_conv2 = DynamicAttention2D(in_planes=self.hidden_size, out_planes=output_dim, grid_shape=grid_shape, kernel_size=3,
                                              ratios=1/16., temperature=temperature, groups=1)
        self.relu = nn.ReLU(inplace=True)

    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)

    def forward(self, x):
        atten1 = self.atten_conv1(x)
        y = self.w1(x, atten1)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        atten2 = self.atten_conv2(y)
        y = self.w2(y, atten2)

        return y

class CNNBlock(nn.Module):
    def __init__(self, hidden_size, grid_shape, padding_mode, p_dropout=0.25, biased=True, temperature=30):
        super(CNNBlock, self).__init__()
        self.hid_size = hidden_size

        self.relu = nn.ReLU(inplace=True)
        self.kernel_size = 3
        conv = TwoBranchDGridConv

        self.w1 = conv(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding_mode=padding_mode, bias=biased)
        self.batch_norm1 = nn.BatchNorm2d(self.hid_size)

        self.w2 = conv(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding_mode=padding_mode, bias=biased)
        self.batch_norm2 = nn.BatchNorm2d(self.hid_size)

        self.atten_conv1 = DynamicAttention2D(in_planes=hidden_size, out_planes=hidden_size, grid_shape=grid_shape, kernel_size=self.kernel_size,
                                              ratios=1 / 16., temperature=temperature, groups=1)
        self.atten_conv2 = DynamicAttention2D(in_planes=hidden_size, out_planes=hidden_size, grid_shape=grid_shape, kernel_size=self.kernel_size,
                                              ratios=1 / 16., temperature=temperature, groups=1)

        self.dropout = nn.Dropout2d(p=p_dropout)


    def forward(self, x):
        atten1 = self.atten_conv1(x)
        y = self.w1(x,atten1)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        atten2 = self.atten_conv2(y)
        y = self.w2(y,atten2)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)


        out = x + y

        return out

class TwoBranchDGridConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode, bias=False):
        super(TwoBranchDGridConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_chn = in_channels
        self.out_chn = out_channels
        self.branch1_weight = nn.Parameter(torch.zeros(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.branch2_weight = nn.Parameter(torch.zeros(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.padding_mode = padding_mode
        if bias:
            self.branch1_bias = nn.Parameter(torch.zeros(out_channels))
            self.branch2_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('branch1_bias', None)
            self.register_parameter('branch2_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight, bias in [[self.branch1_weight, self.branch1_bias], [self.branch2_weight, self.branch2_bias]]:
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(bias, -bound, bound)

    def unfolding_conv(self, x_pad, weight, bias, atten):
        kernel_size = self.branch1_weight.shape[-2]
        batch_size, cin, h_pad, w_pad = x_pad.shape     # B*C*7*7
        h = h_pad - kernel_size + 1
        w = w_pad - kernel_size + 1
        x_unfold = F.unfold(x_pad, (kernel_size, kernel_size))  # B*(C*k*k)*num_block, num_block=5*5
        x_unfold_avg_weight = (x_unfold.reshape(batch_size, cin, kernel_size*kernel_size, h*w) * atten.reshape(batch_size, 1, kernel_size*kernel_size, h*w)).reshape(batch_size, cin*kernel_size*kernel_size, h*w)
        out = x_unfold_avg_weight.transpose(1,2).matmul(weight.view(weight.shape[0], -1).t()).transpose(1,2) # B*(C*k*K)*(5*5)
        out_fold = F.fold(out, (h, w), (1,1))      # B*Cout*5*5
        if bias is not None:
            out_fold = out_fold + bias.reshape(1, -1, 1, 1)

        return out_fold


    def forward(self, x, atten=None):
        pad_size = self.kernel_size // 2
        padding_kwargs = {
            'c':dict(mode='circular'),
            'z':dict(mode='constant', value=0),
            'r':dict(mode='replicate')
        }
        x_branch1 = F.pad(x, [pad_size, pad_size, pad_size, pad_size], **padding_kwargs[self.padding_mode[0]])
        x_branch2 = F.pad(x, [pad_size, pad_size, pad_size, pad_size], **padding_kwargs[self.padding_mode[1]])

        y_branch1 = self.unfolding_conv(x_branch1, weight=self.branch1_weight, bias=self.branch1_bias, atten=atten)
        y_branch2 = self.unfolding_conv(x_branch2, weight=self.branch2_weight, bias=self.branch2_bias, atten=atten)

        out = y_branch1 + y_branch2

        return out

class DynamicAttention2D(nn.Module):
    def __init__(self, in_planes, out_planes, grid_shape, kernel_size, ratios, temperature, init_weight=True,
                 min_channel=16, groups=1):
        super(DynamicAttention2D, self).__init__()
        self.temperature = temperature
        self.kernel_size = kernel_size
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.attention_channel = max(int(in_planes * ratios), min_channel)
        self.out_spatial_size = grid_shape

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, self.attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.attention_channel)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.position_fc = nn.Conv2d(self.attention_channel, self.kernel_size * self.kernel_size * grid_shape[0] * grid_shape[1], 1, bias=True)


        if init_weight:
            self._initialize_weights()

        self.forward_func = self.forward_vanila

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    def forward_vanila(self, x):
        x = self.relu(self.bn(self.fc(self.avgpool(x))))

        x = self.position_fc(x).view(x.size(0), self.kernel_size**2, self.out_spatial_size[0], self.out_spatial_size[1])
        x = self.sigmoid(x/self.temperature)

        return x

    def forward(self, x):
        return self.forward_func(x)






