import torch
from torch import nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, pool):
        super(ConvBlock, self).__init__()

        layers = []
        layers.extend([
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ])
        self.conv = nn.Sequential(*layers)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pool = pool

    def forward(self, x):
        if self.pool:
            x = self.conv(x)
            return self.max_pool(x)
        else:
            return self.conv(x)


class CovPoolFER(nn.Module):
    def __init__(self):
        super(CovPoolFER, self).__init__()

        conv_list = [
            [64, True],
            [96, True],
            [128, False],
            [128, True],
            [256, False],
            [256, False],
        ]

        fc_list = [
            2000,
            128,
            4,  # Finally
        ]

        in_planes = 3
        conv_layers = []
        for c, pool in conv_list:
            conv_layers.append(ConvBlock(in_planes, c, pool))
            in_planes = c

        self.conv_block = nn.Sequential(*conv_layers)

        in_planes = 392*392
        fc_layers = []
        for c in fc_list:
            fc_layers.append(nn.Linear(in_planes, c))
            fc_layers.append(nn.ReLU())
            in_planes = c

        self.fc_block = nn.Sequential(*fc_layers)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate(
            [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to('cuda')
        return torch.index_select(a, dim, order_index)

    def cal_cov_pooling(self, x):
        centers_batch = torch.mean(x.permute(0, 2, 1), 1)
        centers_batch = centers_batch.view(x.shape[0], x.shape[1], 1)
        centers_batch = self.tile(centers_batch, 2, x.shape[2])

        tmp = torch.sub(x, centers_batch)
        tmp_t = tmp.permute(0, 2, 1)
        features_t = 1 / \
            torch.tensor(
                (x.shape[2]-1), dtype=torch.float32).to('cuda')*torch.matmul(tmp_t, tmp)

        trace_temp = []
        for f in features_t:
            trace_temp.append(torch.trace(f))
        trace_t = torch.tensor(trace_temp).to('cuda')

        trace_t = trace_t.view(1, x.shape[0])
        trace_t = self.tile(trace_t, 1, x.shape[1])
        trace_t = 0.0001 * torch.diag(trace_t)
        return torch.add(features_t, trace_t)

    def variable_with_orth_weight_decay(self, shape):
        s1 = torch.tensor(shape[1], dtype=torch.int32).to('cuda')
        s2 = torch.tensor(shape[1]/2, dtype=torch.int32).to('cuda')
        w0_init, _ = torch.qr(torch.normal(0, 1, size=(s1, s2)))
        w0 = torch.nn.Parameter(w0_init).to('cuda')

        tmp1 = w0.view(1, s1, s2)
        tmp2 = w0.transpose(0, 1).view(1, s2, s1)
        tmp1 = self.tile(tmp1, 0, shape[0])
        tmp2 = self.tile(tmp2, 0, shape[0])
        return tmp1, tmp2

    def cal_rect_cov(self, x):
        s, v = torch.symeig(x, eigenvectors=True)
        s = torch.clamp(s, 0.0001, 10000)
        s = torch.diag_embed(s)

        features_t = torch.matmul(torch.matmul(v, s), v.permute(0, 2, 1))

        return features_t

    def cal_log_cov(self, x):
        s, v = torch.symeig(x, eigenvectors=True)
        s = torch.log(s)
        s = torch.diag_embed(s)
        features_t = torch.matmul(torch.matmul(v, s), v.permute(0, 2, 1))

        return features_t

    def forward(self, x):
        x = self.conv_block(x)
        reshaped = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
        local5 = self.cal_cov_pooling(reshaped)

        weight1, weight2 = self.variable_with_orth_weight_decay(local5.shape)
        local6 = torch.matmul(torch.matmul(weight2, local5), weight1)
        local7 = self.cal_rect_cov(local6)
        local9 = self.cal_log_cov(local7)

        local9 = local9.view(local9.shape[0], -1)
        features = self.fc_block(local9)

        return features
