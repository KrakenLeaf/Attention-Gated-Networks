import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3, STNblock, STNblock_v2, UnetConv3_deformable, UnetUp3_CT_deformable
import torch.nn.functional as F
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class unet_t2_ax_cor_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, dataDims=[8,1,96,192,112]):
        super(unet_t2_ax_cor_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.dataDims = dataDims

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Encoder side
        self.encoder_ax = encoder(feature_scale, n_classes, is_deconv, in_channels, # AXIAL
                 nonlocal_mode, attention_dsample, is_batchnorm, dataDims)
        # self.encoder_cor = encoder(feature_scale, n_classes, is_deconv, in_channels, # CORONAL
        #          nonlocal_mode, attention_dsample, is_batchnorm, dataDims)

        # attention blocks
        self.attentionblock2_ax = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3_ax = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4_ax = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock2_cor = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3_cor = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4_cor = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        # self.up_concat4_ax = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        # self.up_concat3_ax = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat4_ax = UnetUp3_CT_deformable(filters[4], filters[3], init_kernel=3, init_padding=1, is_batchnorm=is_batchnorm)
        self.up_concat3_ax = UnetUp3_CT_deformable(filters[3], filters[2], init_kernel=3, init_padding=1, is_batchnorm=is_batchnorm)
        self.up_concat2_ax = UnetUp3_CT(filters[2], filters[1], init_kernel=5, init_padding=2, is_batchnorm=is_batchnorm)
        self.up_concat1_ax = UnetUp3_CT(filters[1], filters[0], init_kernel=7, init_padding=3, is_batchnorm=is_batchnorm)
        # self.up_concat4_cor = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        # self.up_concat3_cor = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat4_cor = UnetUp3_CT_deformable(filters[4], filters[3], init_kernel=3, init_padding=1, is_batchnorm=is_batchnorm)
        self.up_concat3_cor = UnetUp3_CT_deformable(filters[3], filters[2], init_kernel=3, init_padding=1, is_batchnorm=is_batchnorm)
        self.up_concat2_cor = UnetUp3_CT(filters[2], filters[1], init_kernel=5, init_padding=2, is_batchnorm=is_batchnorm)
        self.up_concat1_cor = UnetUp3_CT(filters[1], filters[0], init_kernel=7, init_padding=3, is_batchnorm=is_batchnorm)
        
        # deep supervision
        self.dsv4_ax = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3_ax = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2_ax = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1_ax = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        self.dsv4_cor = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3_cor = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2_cor = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1_cor = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)    
        
        # final conv (without any concat)
        self.final_ax = nn.Conv3d(n_classes*4, n_classes, 1)
        self.final_cor = nn.Conv3d(n_classes*4, n_classes, 1)
        
        # COmbine the outputs from the axial and coronal networks
        self.final_both = nn.Conv3d(n_classes*2, n_classes, 1)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, vol_ax, vol_cor):
        # TODO: Use the same encoder stage for both axial/coronal (i.e. Siamese networks)?
        conv1_ax, conv2_ax, conv3_ax, conv4_ax, center_ax, gating_ax = self.encoder_ax(vol_ax)
        conv1_cor, conv2_cor, conv3_cor, conv4_cor, center_cor, gating_cor = self.encoder_ax(vol_cor)
       
        # Attention Mechanism - Decoder
        # --------------------------------------------------------------------
        # Level 4
        g_conv4_ax, att4_ax = self.attentionblock4_ax(conv4_ax, gating_cor) # Axial w/ coronal gating
        up4_ax = self.up_concat4_ax(g_conv4_ax, center_ax)
        
        g_conv4_cor, att4_cor = self.attentionblock4_cor(conv4_cor, gating_ax) # Coronal w/ axial gating
        up4_cor = self.up_concat4_cor(g_conv4_cor, center_cor)
        
        # Level 3
        g_conv3_ax, att3_ax = self.attentionblock3_ax(conv3_ax, up4_cor) # Axial w/ coronal gating
        up3_ax = self.up_concat3_ax(g_conv3_ax, up4_ax)
        
        g_conv3_cor, att3_cor = self.attentionblock3_cor(conv3_cor, up4_ax) # Coronal w/ axial gating
        up3_cor = self.up_concat3_cor(g_conv3_cor, up4_cor)
        
        # Level 2
        g_conv2_ax, att2_ax = self.attentionblock2_ax(conv2_ax, up3_cor) # Axial w/ coronal gating
        up2_ax = self.up_concat2_ax(g_conv2_ax, up3_ax)
        
        g_conv2_cor, att2_cor = self.attentionblock2_cor(conv2_cor, up3_ax) # Coronal w/ axial gating
        up2_cor = self.up_concat2_cor(g_conv2_cor, up3_cor)

        # Level 1
        up1_ax = self.up_concat1_ax(conv1_ax, up2_ax)
        up1_cor = self.up_concat1_cor(conv1_cor, up2_cor)
    
        # Deep Supervision
        dsv4_ax = self.dsv4_ax(up4_ax) # Axial
        dsv3_ax = self.dsv3_ax(up3_ax)
        dsv2_ax = self.dsv2_ax(up2_ax)
        dsv1_ax = self.dsv1_ax(up1_ax)
        final_ax = self.final_ax(torch.cat([dsv1_ax,dsv2_ax,dsv3_ax,dsv4_ax], dim=1))
        
        dsv4_cor = self.dsv4_cor(up4_cor) # Coronal
        dsv3_cor = self.dsv3_cor(up3_cor)
        dsv2_cor = self.dsv2_cor(up2_cor)
        dsv1_cor = self.dsv1_cor(up1_cor)
        final_cor = self.final_cor(torch.cat([dsv1_cor,dsv2_cor,dsv3_cor,dsv4_cor], dim=1))
        
        # Combined final output
        final_both = self.final_both(torch.cat([final_ax, final_cor], dim=1))
        
        return final_both


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

class encoder(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True, dataDims=[8,1,96,192,112]):
        super(encoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.dataDims = dataDims

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=11, padding_size=15, init_dilation=3)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=9, padding_size=8, init_dilation=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3_deformable(filters[1], filters[2], self.is_batchnorm, kernel_size=5, padding_size=2)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3_deformable(filters[2], filters[3], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3_deformable(filters[3], filters[4], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3, _ = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4, _ = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center, _ = self.center(maxpool4)
        gating = self.gating(center)
        
        return conv1, conv2, conv3, conv4, center, gating


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size, track_running_stats=True),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


