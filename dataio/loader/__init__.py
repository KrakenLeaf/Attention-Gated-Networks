import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset, CMR3DDataset_MultiClass, CMR3DDataset_MultiClass_MultiProj
from dataio.loader.cmr_3D_dataset import CMR3DDataset_MultiClass_MultiProj_unreg, CMR3DDataset_MultiClass_MultiProj_V2
from dataio.loader.cmr_3D_dataset import CMR3DDataset_t2_reg
from dataio.loader._pipeline_dataset import CMR3DDataset_MultiClass_MultiProj_infer

def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'acdc_sax': CMR3DDataset_MultiClass, # OS: Modified class to load multi-labeled datasets
        'mult_prj': CMR3DDataset_MultiClass_MultiProj, # multi label and multi projection (registered)
        'mult_prj_v2': CMR3DDataset_MultiClass_MultiProj_V2, # Multi label and multi-projection
        'mult_prj_unreg': CMR3DDataset_MultiClass_MultiProj_unreg,  # Multi label and multi-projection
        'inference': CMR3DDataset_MultiClass_MultiProj_infer, # Inference mode only
        't2_reg': CMR3DDataset_t2_reg,
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
