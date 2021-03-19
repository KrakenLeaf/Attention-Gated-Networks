from .unet_2D import *
from .unet_3D import *
from .unet_nonlocal_2D import *
from .unet_nonlocal_3D import *
from .unet_grid_attention_3D import *
from .unet_CT_dsv_3D import *
from .unet_CT_single_att_dsv_3D import *
from .unet_CT_multi_att_dsv_3D import *
from .unet_CT_multi_att_dsv_stn_3D import *
from .unet_CT_multi_att_dsv_stn_v2_3D import *
from .unet_CT_multi_att_dsv_stn_unreg_v2_3D import *
from .unet_t2_ax_cor_3D import *
from .unet_CT_dense_multi_att_dsv_3D import *
from .transformer_registration_3D import *
from .unet_CT_multi_att_dsv_deform_3D import *
from .unet_CT_multi_att_dsv_deform_ax_cor_3D import *
from .unet_CT_multi_att_dsv_deform_small_3D import *

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2,2),
                aggregation_mode='concat', dataDims=[4,1,96,192,112], rank=0):
    # NOTE: batchSize is only used for the STN based network. For the other networks, it is irrelevant
    model = _get_model_instance(name, tensor_dim)

    if name in ['unet', 'unet_ct_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False)
    elif name in ['unet_nonlocal']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      is_deconv=False,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale)
    elif name in ['unet_grid_gating',
                  'unet_ct_single_att_dsv',
                  'unet_ct_multi_att_dsv',
                  'unet_ct_dense_multi_att_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False)
    elif name in ['unet_ct_multi_att_dsv_stn',
                  'unet_ct_multi_att_dsv_stn_v2',
                  'unet_ct_multi_att_dsv_stn_unreg_v2',
                  'unet_ct_multi_att_dsv_deform',
                  'unet_ct_multi_att_dsv_deform_small',
                  'unet_CT_multi_att_dsv_deform_ax_cor']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False,
                      dataDims=dataDims).to(rank)
    elif name in ['unet_t2_ax_cor']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False,
                      dataDims=dataDims).to(rank)

        # ToDo: Uncomment the last line for DDP processing - need to do mor eefficiently
        # Currently, we do this locally only for this model
        #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) # Uncomment this for DDP
    elif name in ['transformer_registration']:
        # TODO: Still needs work!
        model = model(vol_size=dataDims[2:],
                      enc_nf=[16, 32, 32, 32],
                      dec_nf=[32, 32, 32, 32, 32, 16, 16])
    else:
        raise 'Model {} not available'.format(name)

    # Utilize multiple GPUs in parallel
    #model = nn.DataParallel(model)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'unet':{'2D': unet_2D, '3D': unet_3D},
        'unet_nonlocal':{'2D': unet_nonlocal_2D, '3D': unet_nonlocal_3D},
        'unet_grid_gating': {'3D': unet_grid_attention_3D},
        'unet_ct_dsv': {'3D': unet_CT_dsv_3D},
        'unet_ct_single_att_dsv': {'3D': unet_CT_single_att_dsv_3D},
        'unet_ct_multi_att_dsv': {'3D': unet_CT_multi_att_dsv_3D}, # Only T2 axial data
        'unet_ct_multi_att_dsv_stn': {'3D': unet_CT_multi_att_dsv_stn_3D}, # Multimodal + Multi-projection
        'unet_ct_multi_att_dsv_stn_v2': {'3D': unet_CT_multi_att_dsv_stn_v2_3D}, # Multimodal via STN blocks
        'unet_ct_multi_att_dsv_stn_unreg_v2': {'3D': unet_CT_multi_att_dsv_stn_unreg_v2_3D}, # Multimodal unregistered via STN blocks
        'unet_ct_dense_multi_att_dsv': {'3D': unet_CT_dense_multi_att_dsv_3D}, # Dense Unet for T2 axial only - doesn't work
        'unet_ct_multi_att_dsv_deform': {'3D': unet_CT_multi_att_dsv_deform_3D},  # Multimodal, deformable convolutions
        'unet_ct_multi_att_dsv_deform_small': {'3D': unet_CT_multi_att_dsv_deform_small_3D},  # Multimodal, deformable convolutions
        'unet_t2_ax_cor': {'3D': unet_t2_ax_cor_3D}, # Axial and Coronal segmentation
        'unet_CT_multi_att_dsv_deform_ax_cor': {'3D': unet_CT_multi_att_dsv_deform_ax_cor_3D}, # Axial and Coronal segmentation - concat along channel dim
        #'unet_t2_ax_cor_deform': {'3D': unet_t2_ax_cor_deform_3D},  # Axial and Coronal segmentation, deformable convs
        'transformer_registration': {'3D': transformer_registration_3D} # T2 axial - coronal registration
    }[name][tensor_dim]
