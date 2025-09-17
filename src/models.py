import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        conv_kernel_sizes=[(2, 3, 3), (2, 3, 3)],
        conv_channels=[32, 64],
        pool_kernel_sizes=[(1, 2, 2), None],
        p_drop_conv=0,
        p_drop_fc=0,
        activation=F.relu,
        approximate=None
    ):
        super(CNN3D, self).__init__()

        assert len(conv_kernel_sizes) == len(conv_channels), \
            "conv_kernel_sizes and conv_channels must have same length"
        assert len(pool_kernel_sizes) == len(conv_channels), \
            "pool_kernel_sizes and conv_channels must have same length"

        self.layers = nn.ModuleList()
        in_ch = input_channels

        for i, (k_size, out_ch, pool_k) in enumerate(zip(conv_kernel_sizes, conv_channels, pool_kernel_sizes)):
            # Convolution
            conv = nn.Conv3d(in_ch, out_ch, kernel_size=k_size,
                             padding=tuple(k // 2 for k in k_size))  # padding to keep dims
            bn = nn.BatchNorm3d(out_ch)
            drop = nn.Dropout3d(p_drop_conv)
            pool = nn.MaxPool3d(kernel_size=pool_k) if pool_k is not None else None

            self.layers.append(nn.ModuleDict({
                "conv": conv,
                "bn": bn,
                "drop": drop,
                "pool": pool
            }))
            in_ch = out_ch  # update for next layer

        # Global pooling + FC head
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.drop_fc = nn.Dropout(p_drop_fc)
        self.fc = nn.Linear(in_ch, output_dim)

        # Activation
        if activation == F.gelu and approximate is not None:
            self.activation = lambda x: activation(x, approximate=approximate)
        else:
            self.activation = activation

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)  # (batch, channels, tod, lat, lon)

        for layer in self.layers:
            x = layer["bn"](self.activation(layer["conv"](x)))
            x = layer["drop"](x)
            if layer["pool"] is not None:
                x = layer["pool"](x)

        x = self.global_pool(x)  # (batch, channels, 1, 1, 1)
        x = self.flatten(x)      # (batch, channels)
        x = self.drop_fc(x)
        return self.fc(x)