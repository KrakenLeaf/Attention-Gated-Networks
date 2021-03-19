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
    dataset = dataset_class(dataset_path, split='test_val', transform=dataset_transform['valid']) # 'validation'
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

        try:
            input_arr = np.squeeze(data[0].cpu().numpy()).astype(np.float32)
        except:
            input_arr = np.squeeze(data[0][0].cpu().numpy()).astype(np.float32)
        label_arr = np.squeeze(data[1].cpu().numpy()).astype(np.int16)
        output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)
        logit_arr = np.squeeze(model.logits.data.cpu().numpy()).astype(np.float32)

        #print("output_arr shape: {}".format(output_arr.shape))

        # Clean the prediction
        output_arr = remove_islands_gp(output_arr)

        # If there is a label image - compute statistics
        #dice_vals = dice_score(label_arr, output_arr, n_class=int(json_opts.model.output_nc)) # DICE - left and right together
        dice_vals, dice_vals_left_right = dice_score_average_left_right(label_arr, output_arr, n_class=int(json_opts.model.output_nc))  # DICE - av. of left + right
        print("DICE scores: {}".format(dice_vals_left_right))

        # hd, msd = distance_metric_new(label_arr, output_arr, vox_size=0.390625) # Hausdorff distance and mean surface distance
        hd, msd = distance_metric_new_left_right(label_arr, output_arr, vox_size=0.390625)  # Hausdorff distance and mean surface distance

        volumes = calculate_volume(output_arr, pixdim=[0.390625]) # Segmentation volume
        volumes_label = calculate_volume(label_arr, pixdim=[0.390625])  # Segmentation volume
        cm_dist, cm_diff = measure_cm_dist_wrapper(output_arr, label_arr, pixdim=[0.390625]) # Center of mass difference
        precision, recall = precision_and_recall(label_arr, output_arr, n_class=int(json_opts.model.output_nc)) # Precision and recall

        # Accumulate stats
        stat_logger.update(split='test', input_dict={'img_name': iteration,
                                                     'Background': dice_vals[0],
                                                     'GPe_dice left': dice_vals_left_right[0, 0],
                                                     'GPe_dice right': dice_vals_left_right[0, 1],
                                                     'GPi_dice left': dice_vals_left_right[1, 0],
                                                     'GPi_dice right': dice_vals_left_right[1, 1],
                                                     'GPe_hd left [mm]': hd[0, 0],
                                                     'GPe_hd right [mm]': hd[0, 1],
                                                     'GPi_hd left [mm]': hd[1, 0],
                                                     'GPi_hd right [mm]': hd[1, 1],
                                                     'GPe_msd left [mm]': msd[0, 0],
                                                     'GPe_msd right [mm]': msd[0, 1],
                                                     'GPi_msd left [mm]': msd[1, 0],
                                                     'GPi_msd right [mm]': msd[1, 1],
                                                     'GPe right vol [cm^3]': volumes[0] / 1000,
                                                     'GPe left vol [cm^3]': volumes[1] / 1000,
                                                     'GPi right vol [cm^3]': volumes[2] / 1000,
                                                     'GPi left vol [cm^3]': volumes[3] / 1000,
                                                     'GPe right label vol [cm^3]': volumes_label[0] / 1000,
                                                     'GPe left label vol [cm^3]': volumes_label[1] / 1000,
                                                     'GPi right label vol [cm^3]': volumes_label[2] / 1000,
                                                     'GPi left label vol [cm^3]': volumes_label[3] / 1000,
                                                     'GPe right cm dist [mm]': cm_dist[0, 0],
                                                     'GPe left cm dist [mm]': cm_dist[0, 1],
                                                     'GPe right cm X [mm]': cm_diff[0][0][0],
                                                     'GPe right cm Y [mm]': cm_diff[0][0][1],
                                                     'GPe right cm Z [mm]': cm_diff[0][0][2],
                                                     'GPe left cm X [mm]': cm_diff[0][1][0],
                                                     'GPe left cm Y [mm]': cm_diff[0][1][1],
                                                     'GPe left cm Z [mm]': cm_diff[0][1][2],
                                                     'GPi right cm dist [mm]': cm_dist[1, 0],
                                                     'GPi left cm dist [mm]': cm_dist[1, 1],
                                                     'GPi right cm X [mm]': cm_diff[1][0][0],
                                                     'GPi right cm Y [mm]': cm_diff[1][0][1],
                                                     'GPi right cm Z [mm]': cm_diff[1][0][2],
                                                     'GPi left cm X [mm]': cm_diff[1][1][0],
                                                     'GPi left cm Y [mm]': cm_diff[1][1][1],
                                                     'GPi left cm Z [mm]': cm_diff[1][1][2],
                                                     'GPe_prec': precision[1],
                                                     'GPe_reca': recall[1],
                                                     'GPi_prec': precision[2],
                                                     'GPi_reca': recall[2],
                                                     })

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
        default='configs/config_unet_only_gp.json')
    args = parser.parse_args()

    validation(args.config)
