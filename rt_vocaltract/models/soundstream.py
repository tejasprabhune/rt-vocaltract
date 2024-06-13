import torch

class SoundStreamInversion(torch.nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 strides: list, 
                 dilations: list):
        super(SoundStreamInversion, self).__init__()


        self.in_conv = torch.nn.Conv1d(in_channels=in_channels,
                                       out_channels=out_channels, 
                                       kernel_size=kernel_size, 
                                       stride=1)
                                    
        strides.insert(0, 1)

        # Can be made member later
        channel_multipliers: list = [1, 2, 4, 8, 16]
        self.encoder_blocks: torch.nn.ModuleList = torch.nn.ModuleList([
            EncoderBlock(
                in_channels=out_channels * channel_multipliers[i - 1],
                out_channels=out_channels * channel_multipliers[i],
                stride=strides[i], 
                dilations=dilations, 
                kernel_size=kernel_size
                )
            for i in range(1, len(strides))
        ])
        
        self.out_conv: torch.nn.Conv1d = \
            torch.nn.Conv1d(in_channels=out_channels * channel_multipliers[-1],
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1)

    def forward(self, x):
        x = self.in_conv(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.out_conv(x)
        return x

class ResidualUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation):
        super(ResidualUnit, self).__init__()
        self.dilated_conv = torch.nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            dilation=dilation)
        self.single_conv = torch.nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=stride)
    
    def forward(self, x):
        conv_x = self.dilated_conv(x)
        conv_x = self.single_conv(conv_x)
        x = x[:, :, :conv_x.shape[2]]
        return x + conv_x

class EncoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride, 
                 dilations,
                 kernel_size):
        super(EncoderBlock, self).__init__()
        self.res_units = torch.nn.ModuleList([
            ResidualUnit(
                in_channels=in_channels, 
                out_channels=out_channels // 2, 
                kernel_size=kernel_size, 
                stride=1, 
                dilation=dilation
                )
            for dilation in dilations
        ])
        self.conv = torch.nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=2 * stride, 
            stride=stride)
    
    def forward(self, x):
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv(x)
        return x