import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from torch.utils.data import DataLoader
from torch import load

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
import re

from models import get_model
import numpy as np
import os
from utils.error_logger import StatLogger
from tqdm import tqdm
from utils.metrics import dice_score, dice_score_average_left_right
from utils.metrics import distance_metric
from utils.metrics import precision_and_recall
from utils.metrics import distance_metric_new, distance_metric_new_left_right
from utils.measure_registration_results import *

from scipy.special import logsumexp

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def validation(json_name):
    # Load options
    json_opts = json_file_to_pyobj(json_name)
    train_opts = json_opts.training

    # Should be consistent with what was used as input to train the network
    model_types = train_opts.modalities

    # Setup Dataset and Augmentation
    dataset_class = get_dataset(train_opts.arch_type)
    dataset_path = get_dataset_path(train_opts.arch_type, json_opts.data_path)
    dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)

    # Setup Data Loader
    dataset = dataset_class(dataset_path, split='test_val', transform=dataset_transform['valid'], modalities=train_opts.modalities) # 'validation'
    data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=False)

    # Setup the NN Model
    dataDims = [1, 1, dataset.image_dims[0], dataset.image_dims[1],
                dataset.image_dims[2]]
    model = get_model(json_opts.model, dataDims)
    save_directory = os.path.join(model.save_dir, train_opts.arch_type)
    mkdirfun(save_directory)

    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)

    # Setup stats logger
    stat_logger = StatLogger()

    # test
    for iteration, data in enumerate(data_loader, 1):
        identifier = data[2].cpu().numpy()[0]
        #pname = identifier2id(identifier, json_opts.data_path[0])
        #print("Iteration {}, patient {}".format(iteration, pname))
        print("Iteration {}".format(iteration))

        model.set_input(data[0], data[1])
        model.test()

        #print("data shape {}".format(data[0].size))
        #print("logits shape {}".format(model.prediction.shape))

        try:
            input_arr = np.squeeze(data[0].cpu().numpy()).astype(np.float32)
        except:
            input_arr = np.squeeze(data[0][0].cpu().numpy()).astype(np.float32)
        label_arr = np.squeeze(data[1].cpu().numpy()).astype(np.int16)
        output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)
        logit_arr = np.squeeze(model.prediction.data.cpu().numpy()).astype(np.float32)
        Energy = free_enrgy(logit_arr, Temp=1)
        #print("Free energy dimensions = {}".format(Energy.shape))

        # Clean the prediction
        output_arr = remove_islands(output_arr)

        do_statistics_flag = 1
        if do_statistics_flag == 1:
            # If there is a label image - compute statistics
            #dice_vals = dice_score(label_arr, output_arr, n_class=int(json_opts.model.output_nc)) # DICE - left and right together
            dice_vals, dice_vals_left_right = dice_score_average_left_right(label_arr, output_arr, n_class=int(json_opts.model.output_nc))  # DICE - av. of left + right
            print("DICE scores: {}".format(dice_vals_left_right))

            if 0.0 in dice_vals_left_right:
                hd = -1.0 * np.ones(dice_vals_left_right.shape)
                msd = -1.0 * np.ones(dice_vals_left_right.shape)
                volumes = [-1.0 for f in range(2 * (json_opts.model.output_nc - 1))]
                volumes_label = calculate_volume(label_arr, pixdim=[0.390625])  # Segmentation volume
                cm_dist = -1.0 * np.ones(dice_vals_left_right.shape)
                precision = [-1.0 for f in range(json_opts.model.output_nc)]
                recall = [-1.0 for f in range(json_opts.model.output_nc)]
            else:
                # hd, msd = distance_metric_new(label_arr, output_arr, vox_size=0.390625) # Hausdorff distance and mean surface distance
                hd, msd = distance_metric_new_left_right(label_arr, output_arr, vox_size=0.390625)  # Hausdorff distance and mean surface distance
                volumes = calculate_volume(output_arr, pixdim=[0.390625]) # Segmentation volume
                volumes_label = calculate_volume(label_arr, pixdim=[0.390625])  # Segmentation volume
                cm_dist, cm_diff = measure_cm_dist_wrapper(output_arr, label_arr, pixdim=[0.390625]) # Center of mass difference
                precision, recall = precision_and_recall(label_arr, output_arr, n_class=int(json_opts.model.output_nc)) # Precision and recall


            # Accumulate stats
            stat_logger.update(split='test', input_dict={'img_name': iteration,
                                                         'Background': dice_vals[0],
                                                         'stn_dice left': dice_vals_left_right[0, 0],
                                                         'stn_dice right': dice_vals_left_right[0, 1],
                                                         'sn_dice left': dice_vals_left_right[1, 0],
                                                         'sn_dice right': dice_vals_left_right[1, 1],
                                                         'rn_dice left': dice_vals_left_right[2, 0],
                                                         'rn_dice right': dice_vals_left_right[2, 1],
                                                         # ------
                                                         'stn_hd left [mm]': hd[0, 0],
                                                         'stn_hd right [mm]': hd[0, 1],
                                                         'sn_hd left [mm]': hd[1, 0],
                                                         'sn_hd right [mm]': hd[1, 1],
                                                         'rn_hd left [mm]': hd[2, 0],
                                                         'rn_hd right [mm]': hd[2, 1],
                                                         # ------
                                                         'stn_msd left [mm]': msd[0, 0],
                                                         'stn_msd right [mm]': msd[0, 1],
                                                         'sn_msd left [mm]': msd[1, 0],
                                                         'sn_msd right [mm]': msd[1, 1],
                                                         'rn_msd left [mm]': msd[2, 0],
                                                         'rn_msd right [mm]': msd[2, 1],
                                                         # ------
                                                         'stn right vol [cm^3]': volumes[0] / 1000,
                                                         'stn left vol [cm^3]': volumes[1] / 1000,
                                                         'sn right vol [cm^3]': volumes[2] / 1000,
                                                         'sn left vol [cm^3]': volumes[3] / 1000,
                                                         'rn right vol [cm^3]': volumes[4] / 1000,
                                                         'rn left vol [cm^3]': volumes[5] / 1000,
                                                         'stn right label vol [cm^3]': volumes_label[0] / 1000,
                                                         'stn left label vol [cm^3]': volumes_label[1] / 1000,
                                                         'sn right label vol [cm^3]': volumes_label[2] / 1000,
                                                         'sn left label vol [cm^3]': volumes_label[3] / 1000,
                                                         'rn right label vol [cm^3]': volumes_label[4] / 1000,
                                                         'rn left label vol [cm^3]': volumes_label[5] / 1000,
                                                         # ------
                                                         'stn right cm dist [mm]': cm_dist[0, 0],
                                                         'stn left cm dist [mm]': cm_dist[0, 1],
                                                         'sn right cm dist [mm]': cm_dist[1, 0],
                                                         'sn left cm dist [mm]': cm_dist[1, 1],
                                                         'rn right cm dist [mm]': cm_dist[2, 0],
                                                         'rn left cm dist [mm]': cm_dist[2, 1],
                                                         # ------
                                                         'stn_prec': precision[1],
                                                         'stn_reca': recall[1],
                                                         'sn_prec': precision[2],
                                                         'sn_reca': recall[2],
                                                         'rn_prec': precision[3],
                                                         'rn_reca': recall[3],
                                                         })

        # Save files to disc
        # ------------------------------------------------------------------------------------

        # Write a nifti image
        import SimpleITK as sitk
        if input_arr.ndim <= 3:
            input_img = sitk.GetImageFromArray(np.transpose(input_arr, (2, 1, 0))); input_img.SetDirection([-1,0,0,0,-1,0,0,0,1]) # Original
        else:
            input_arr = np.squeeze(input_arr)#; input_arr = np.squeeze(input_arr[0, :, :, :])

        # Save labels and predictions
        label_img = sitk.GetImageFromArray(np.transpose(label_arr, (2, 1, 0)))
        label_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        predi_img = sitk.GetImageFromArray(np.transpose(output_arr, (2, 1, 0)))
        predi_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        sitk.WriteImage(label_img, os.path.join(save_directory, '{}_lbl.nii.gz'.format(identifier)))
        sitk.WriteImage(predi_img, os.path.join(save_directory, '{}_pred.nii.gz'.format(identifier)))

        # Save the logits - probability maps for each class
        # for qq in range(int(json_opts.model.output_nc)):
        #     logit_img = sitk.GetImageFromArray(np.transpose(np.squeeze(logit_arr[qq, ...]), (2, 1, 0))) #
        #     logit_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        #     sitk.WriteImage(logit_img, os.path.join(save_directory, '{}_logit_class_{}.nii.gz'.format(iteration, qq)))

        # Save the energy
        logit_img = sitk.GetImageFromArray(np.transpose(Energy, (2, 1, 0)))  #
        logit_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        sitk.WriteImage(logit_img, os.path.join(save_directory, '{}_energy_class.nii.gz'.format(identifier)))

        #print("iteration: {}".format(iteration))
        for ii in range(len(model_types)):
            #print("{}".format(input_arr.shape))
            try:
                input_img = sitk.GetImageFromArray(np.transpose(np.squeeze(input_arr[ii, :, :, :]), (2, 1, 0)))
            except:
                input_img = sitk.GetImageFromArray(np.transpose(np.squeeze(input_arr), (2, 1, 0)))
            input_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1]) # For multimodal
            sitk.WriteImage(input_img, os.path.join(save_directory,'{}_img_{}.nii.gz'.format(identifier, model_types[ii])))


    stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(save_directory,'stats.csv'))
    for key, (mean_val, std_val) in stat_logger.get_errors(split='test').items():
        print('-',key,': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val),'-')

# Calculate Helmholtz (negative) free energy
def free_enrgy(input, Temp):
    #print("Input dims = {}".format(input.shape))
    E = Temp * logsumexp((input / Temp), axis=0)
    #print("E dims = {}".format(E.shape))
    return np.squeeze(E)

# Attach correct patient ID to identifier
def identifier2id(identifier, db_path):
    # Open file and look for patient according to identifier
    with open(os.path.join(db_path, 'Config.txt')) as file:
        # Get all lines
        lines = file.readlines()

        # Go over the lines
        name = []
        counter = 0
        for line in lines:
            if counter > 3: # Start from line 4
                #print("line - {}".format(line))
                tmp = int(re.search(',(.*)', line).group(1)) # Found identifier number
                if identifier == tmp:
                    name = re.search('(.*),', line).group(1) # Found name
                    break

            counter += 1

    return name

# Workaround to run code from debug
class InputArgs:
    def __init__(self, json_file, debug):
        self.config = json_file
        self.debug = debug

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Validation Function')
    parser.add_argument('-c', '--config', 
        help='testing config file', 
        default='configs/config_unet_only_stn_sn_rn.json')
    args = parser.parse_args()

    validation(args.config)
