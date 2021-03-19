import copy
import csv
import itertools
import os

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io
from scipy import ndimage
from skimage import measure
from stl import mesh
from copy import deepcopy



#from sispipeline.utils.dirservice import *
#from sispipeline.utils.logservice import *
#from sistools import spc


def _load(path):
    if path.endswith('.mat'):
        mat_data = scipy.io.loadmat(path)
        result = mat_data['temp'].swapaxes(0, 1)
        header = -1
    else:
        struct = nib.load(path)
        result = struct.get_fdata()
        header = struct.header

    return result, header

def check_cc_of_same_object(img_A, img_B, class_id=-1):
    '''
    TESTED ONLY FOR TWO CCs

    Helper function: checks if the cc algorithm labeled in both A and B the same structures with the same labels.
    This code handles the case were there are two CCs in the image - left and right parts of the brain
    :param img_A: 3d array
    :param img_B: 3d array of same size as img_A
    :return: True/False flag - if True then there is some overlap
    '''
    # Number of cc classes
    n_classes = np.max(img_A)

    # Calculate center of mass for each image
    com_A = np.zeros((n_classes, 3))
    for ii in range(n_classes):
        struct1 = copy.deepcopy(img_A)
        struct1[struct1 != ii + 1] = 0
        struct1[struct1 != 0] = 1
        com_A[ii, :] = np.array(ndimage.measurements.center_of_mass(struct1))

    com_B = np.zeros((n_classes, 3))
    for ii in range(n_classes):
        struct1 = copy.deepcopy(img_B)
        struct1[struct1 != ii + 1] = 0
        struct1[struct1 != 0] = 1
        com_B[ii, :] = np.array(ndimage.measurements.center_of_mass(struct1))

    # Calculate distance norms
    norm1 = np.linalg.norm(com_A[0, :] - com_B[0, :]) # Distance of presumably same cc
    norm2 = np.linalg.norm(com_A[0, :] - com_B[1, :]) # Distance between the "opposite" ccs

    if norm2 < norm1:
        # In this case we need to switch CCs
        img_A[img_A == 1] = 3
        img_A[img_A == 2] = 1
        img_A[img_A == 3] = 2
        print("Flipped CCs in img_A, class ID: {}".format(class_id))

    return img_A

# Calculate segmentation volume
def calculate_volume(vol, pixdim=1, max_classes=2, dtype=np.uint8):
    import cc3d  # pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)
    from copy import deepcopy

    vol = vol.astype(dtype)

    if len(pixdim) > 1:
        voxel_size = [pixdim[0], pixdim[1], pixdim[2]]
    else:
        voxel_size = [pixdim[0], pixdim[0], pixdim[0]]

    # Voxel volume in mm^3
    voxel_size_mm3 = voxel_size[0] * voxel_size[1] * voxel_size[2]

    # Work per each segmentatino class
    vox_count_mm3 = []
    for ii in range(1, np.max(vol) + 1):
        # Take class
        vol_use = deepcopy(vol)
        vol_use[vol_use != ii] = 0
        vol_use[vol_use != 0] = 1

        # Divide into left and right using connected components
        labels_out = cc3d.connected_components(vol_use.astype(dtype))  # 26-connected (default)

        # I added this to be a parameter, because in P034 cc3d accidentally marked ~21 pixels as a third class
        labels_out[labels_out > max_classes] = max_classes

        # Extract individual components
        for segid in range(1, np.max(labels_out) + 1):
            extracted_image = labels_out * (labels_out == segid)
            extracted_image[extracted_image != 0] = 1

            # Calculate number of voxels
            vox_count_tmp = np.sum(extracted_image[extracted_image != 0])
            vox_count_mm3.append(vox_count_tmp * voxel_size_mm3)

    return vox_count_mm3

# Remove unwanted segmentation islands - only tested for GP (both sides)
def remove_islands_gp(seg_vol, max_islands_to_keep=2, dtype=np.uint8):
    import cc3d #  pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)
    from copy import deepcopy

    seg_vol_orig = deepcopy(seg_vol).astype(dtype)

    # Create components
    seg_vol[seg_vol != 0] = 1
    labels_out = cc3d.connected_components(seg_vol.astype(dtype)) # 26-connected (default)

    # Extract individual components
    vox_count = []
    extracted_image_buffer = []
    N = np.max(labels_out)
    for segid in range(1, N+1):
        extracted_image = labels_out * (labels_out == segid)
        #extracted_image[extracted_image != 0] = 1
        extracted_image_buffer.append(extracted_image.astype(dtype))

        # Calculate number of voxels
        vox_count_tmp = np.sum(extracted_image[extracted_image != 0])
        vox_count.append(vox_count_tmp)

    # For GP (i+e) keep only the two largest islands
    smallest_indices = sorted(range(len(vox_count)), key=lambda x: vox_count[x])[0:len(vox_count) - max_islands_to_keep]
    for ii in range(len(smallest_indices)):
        mask = extracted_image_buffer[smallest_indices[ii]]
        mask[mask != 0] = 1
        mask = 1 - mask
        seg_vol_orig = seg_vol_orig * mask

    return seg_vol_orig

# Remove unwanted segmentation islands
def remove_islands(seg_vol, max_islands_to_keep=2, max_label_num=3, dtype=np.uint8):
    import cc3d #  pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)
    from copy import deepcopy

    # Voxels with values larger than max_label_num are (arbitrarily) assigned to 0
    seg_vol[seg_vol > max_label_num] = 0

    seg_vol = seg_vol.astype(dtype)

    # Deep copy
    seg_vol_orig = deepcopy(seg_vol).astype(dtype)

    # Extract individual components
    N = max_label_num
    for segid in range(1, N+1): # For each class
        vox_count = []
        extracted_image_buffer = []

        # Extract class
        extracted_image = seg_vol * (seg_vol == segid)

        # Calculate connected components
        #extracted_image[extracted_image != 0] = 1
        cc_labels = cc3d.connected_components(extracted_image.astype(dtype))  # 26-connected (default)

        for qq in range(1, np.max(cc_labels)+1):
            # Accumulate
            extracted_image_2 = cc_labels * (cc_labels == qq)
            extracted_image_buffer.append(extracted_image_2.astype(dtype))

            # Calculate number of voxels
            vox_count_tmp = np.sum(extracted_image_2[extracted_image_2 != 0]) / qq # Normalize the value of the non-zero voxels
            vox_count.append(vox_count_tmp)

        # For each class: keep only the two largest islands
        smallest_indices = sorted(range(len(vox_count)), key=lambda x: vox_count[x])[0:len(vox_count) - max_islands_to_keep]
        for ii in range(len(smallest_indices)):
            mask = extracted_image_buffer[smallest_indices[ii]]
            mask[mask != 0] = 1
            mask = 1 - mask
            seg_vol_orig = seg_vol_orig * mask

    # Recombine all corrected classes

    return seg_vol_orig

def divide_pred_by_class(seg_vol, dtype=np.int8):
    seg_vol = seg_vol.astype(dtype) # Cast as integers

    volumes = []
    for ii in range(np.max(seg_vol)):
        tmp = deepcopy(seg_vol)
        tmp[tmp != ii + 1] = 0
        tmp[tmp > 0] = 1
        volumes.append(tmp)

    return volumes

# Wrapper function for calculating the center of mass difference
def measure_cm_dist_wrapper(struct1, struct2, pixdim=1):
    struct1_by_class = divide_pred_by_class(struct1)
    struct2_by_class = divide_pred_by_class(struct2)

    cm_dist = []
    cm_diff = []
    for ii in range(len(struct1_by_class)):
        tmp_cm_dist, tmp_cm_diff = measure_CM_dist_real_size(struct1_by_class[ii], struct2_by_class[ii], pixdim)
        cm_dist.append(tmp_cm_dist)
        cm_diff.append(tmp_cm_diff)

    '''
    Output format, e.g. for GP:
    cm_dist: GPe right cm_dist[0, 0], GPe left cm_dist[0, 1]
             GPi right cm_dist[1, 0], GPi left cm_dist[1, 1]
    cm_diff: GPe right cm_diff[0][0], GPe left cm_diff[0][1]
             GPi right cm_diff[1][0], GPi left cm_diff[1][1]
    '''
    return np.array(cm_dist) , np.array(cm_diff)

def measure_CM_dist(struct1, struct2):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :return: Euclidan distance between center of mass of the structures (in voxels)
    """

    # Compute cm for all labels greater than 0
    struct1_cm = np.array(ndimage.measurements.center_of_mass(struct1))
    struct2_cm = np.array(ndimage.measurements.center_of_mass(struct2))
    cm_diff = struct1_cm - struct2_cm
    cm_dist = np.linalg.norm(cm_diff)
    return cm_dist, cm_diff


def measure_CM_dist_real_size(struct1, struct2, pixdim=1, dtype=np.uint8, max_classes=2):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim - should be a list, even of one element (if isotropic resolution):
    :param max_classes - Should be 2: corresponds to left and right
    :return: Euclidan distance between center of mass of the structures (in mm)
    """
    import cc3d  # pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)

    if len(pixdim) > 1:
        voxel_size = [pixdim[2], pixdim[1], pixdim[3]] # OS: This looks strange...
    else:
        voxel_size = [pixdim[0], pixdim[0], pixdim[0]]

    # Construct labels - connected components
    #lbl1 = ndimage.label(struct1.astype(dtype))[0]# Looks like there's a bug here...
    #lbl2 = ndimage.label(struct2.astype(dtype))[0]

    lbl1 = cc3d.connected_components(struct1.astype(dtype))  # 26-connected (default)
    lbl2 = cc3d.connected_components(struct2.astype(dtype))  # 26-connected (default)

    # cc3d might output a pixel as a nex class - observed in PD55.nii.gz for DISTAL 2017 - single pixel=3
    lbl1[lbl1 > max_classes] = max_classes
    lbl2[lbl2 > max_classes] = max_classes

    lbl1 = check_cc_of_same_object(lbl1, lbl2)

    lbl_index = [f for f in range(1, np.max(lbl1) + 1)]

    struct1_cm = np.array(ndimage.measurements.center_of_mass(struct1, lbl1, lbl_index))
    struct2_cm = np.array(ndimage.measurements.center_of_mass(struct2, lbl2, lbl_index))

    cm_diff = []
    cm_dist = []
    for ii in range(len(struct1_cm)):
        cm_diff_tmp = (struct1_cm[ii] - struct2_cm[ii]) * voxel_size
        cm_dist_tmp = np.linalg.norm(cm_diff_tmp)
        cm_diff.append(cm_diff_tmp)
        cm_dist.append(cm_dist_tmp)

    return cm_dist, cm_diff


def measure_surface_dist(struct1, struct2):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :return: Averaged Euclidan distance between the structures surface points (in voxels)
    """
    verts1, faces1, normals1, values1 = measure.marching_cubes(struct1, 0.5)
    verts2, faces2, normals2, values2 = measure.marching_cubes(struct2, 0.5)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm((verts2 - surface_point), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def measure_surface_dist_stl(struct1, struct2):
    """
    :param struct1: stl filename
    :param struct2: stl filename
    :return: Averaged Euclidan distance between the structures surface points (in voxels)
    """
    mesh1 = mesh.Mesh.from_file(struct1)
    mesh2 = mesh.Mesh.from_file(struct2)
    verts1 = np.concatenate((mesh1.v0, mesh1.v1, mesh1.v2), axis=0)
    verts2 = np.concatenate((mesh2.v0, mesh2.v1, mesh2.v2), axis=0)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm((verts2 - surface_point), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def measure_surface_dist_real_size(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim:
    :return: Averaged Euclidan distance between the structures surface points (in mm)
    """
    voxel_size = [pixdim[2], pixdim[1], pixdim[3]]

    verts1, faces1, normals1, values1 = measure.marching_cubes(struct1, 0.5)
    verts2, faces2, normals2, values2 = measure.marching_cubes(struct2, 0.5)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm(((verts2 - surface_point) * voxel_size), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def measure_dc(struct1, struct2):
    a = struct1.astype(bool)
    b = struct2.astype(bool)
    # noinspection PyTypeChecker
    dice = float(2.0 * np.sum(a * b)) / (np.sum(a) + np.sum(b))
    # dice_dist = np.sum(a[b==1])*2.0 / (np.sum(a) + np.sum(b))

    return dice


def measure_cdc(struct1, struct2):

    size_of_A_intersect_B = np.sum(struct1 * struct2)
    size_of_A = np.sum(struct1)
    size_of_B = np.sum(struct2)
    if size_of_A_intersect_B > 0:
        c = size_of_A_intersect_B/np.sum(struct1 * np.sign(struct2))
    else:
        c = 1.0
    cdc = (2.0*size_of_A_intersect_B) / (c*size_of_A + size_of_B)

    return cdc


def measure_prior_reg_results(path_to_prior_reg_lst, path_to_most_similar_lst, threshold=0):
    """
    :param path_to_prior_reg_lst: List of paths to the priors registered to the clinical patient space (RN, SN, STN)
    :param path_to_most_similar_lst: List of the structures of most similar brain
    :param threshold:
    :return: Array of the CM distance and mean surface distance of the prior from most similar
    """
    reg_results_cm = np.zeros(len(path_to_prior_reg_lst))
    reg_results_surface = np.zeros(len(path_to_prior_reg_lst))
    cm_diff_result = np.zeros([len(path_to_prior_reg_lst), 3])
    reg_results_dice = np.zeros(len(path_to_prior_reg_lst))
    for ind, prior_path in enumerate(path_to_prior_reg_lst):
        gt_path = path_to_most_similar_lst[ind]
        gt_data = _load(gt_path)
        prior_data = _load(prior_path)

        if threshold > 0:
            gt_data[gt_data < threshold] = 0
            prior_data[prior_data < threshold] = 0

        reg_results_cm[ind], cm_diff_result[ind, :] = measure_CM_dist(prior_data, gt_data)
        reg_results_surface[ind] = measure_surface_dist(prior_data, gt_data)
        reg_results_dice[ind] = measure_dc(prior_data, gt_data)

    return reg_results_cm, cm_diff_result, reg_results_surface, reg_results_dice


def measure_prior_reg_results_real_size(path_to_prior_reg_lst, path_to_most_similar_lst, threshold=0):
    """
    :param path_to_prior_reg_lst: List of paths to the priors registered to the clinical patient space (RN, SN, STN)
    :param path_to_most_similar_lst: List of the structures of most similar brain
    :param threshold:
    :return: Array of the CM distance and mean surface distance of the prior from most similar
    """
    reg_results_cm = np.zeros(len(path_to_prior_reg_lst))
    reg_results_surface = np.zeros(len(path_to_prior_reg_lst))
    cm_diff_result = np.zeros([len(path_to_prior_reg_lst), 3])
    reg_results_dice = np.zeros(len(path_to_prior_reg_lst))
    for ind, prior_path in enumerate(path_to_prior_reg_lst):
        gt_path = path_to_most_similar_lst[ind]
        gt_data = _load(gt_path)
        prior_data = _load(prior_path)

        if threshold > 0:
            gt_data[gt_data < threshold] = 0
            prior_data[prior_data < threshold] = 0
        # pdb.set_trace()
        # if FlowConfig().correct_bias:
        #     prior_data = np.roll(prior_data, -1, axis=2)

        # pixel size from the header information
        gt_img = nib.load(gt_path)
        pixdim = gt_img.header['pixdim']

        reg_results_cm[ind], cm_diff_result[ind, :] = measure_CM_dist_real_size(prior_data, gt_data, pixdim)
        reg_results_surface[ind] = measure_surface_dist_real_size(prior_data, gt_data, pixdim)
        reg_results_dice[ind] = measure_dc(prior_data, gt_data)

    return reg_results_cm, cm_diff_result, reg_results_surface, reg_results_dice


def check_reg_results(context, most_similar_prior, priors_list):
    """

    :param context: 
    :param most_similar_prior:
    :param priors_list: similar priors list
    :return: (1) Data frame of center of mass distance for each prior
             (2) Data frame of average surface points distance for each prior

    """

    patient_reg_results_cm = []
    patient_reg_results_surface = []
    patient_reg_results_dice = []
    path_to_ground_truth_lst = sorted(context.list_files(DSN.PRIOR_REGTO_BET_3D(prior=most_similar_prior,
                                                                                modality=T2_MODALITY,
                                                                                side=ALL,
                                                                                bg_structure=UNMERGED,
                                                                                thr05=True,
                                                                                cropped=True)))

    priors_name_lst = priors_list

    for prior in priors_name_lst:
        path_to_prior_reg_lst = sorted(context.list_files(DSN.PRIOR_REGTO_BET_3D(prior=prior,
                                                                                 modality=T2_MODALITY,
                                                                                 side=ALL,
                                                                                 bg_structure=UNMERGED,
                                                                                 thr05=True,
                                                                                 cropped=True)))
        reg_results_cm, reg_results_cm_diff, reg_results_surface, reg_results_dice = \
            measure_prior_reg_results(path_to_prior_reg_lst, path_to_ground_truth_lst)
        patient_reg_results_cm.append(reg_results_cm)
        patient_reg_results_surface.append(reg_results_surface)
        patient_reg_results_dice.append(reg_results_dice)

    structure_lst = ['RN_L', 'RN_R', 'SN_L', 'SN_R', 'STN_L', 'STN_R']
    patient_reg_results_cm_df = pd.DataFrame(patient_reg_results_cm, index=priors_name_lst, columns=structure_lst)
    patient_reg_results_surface_df = pd.DataFrame(patient_reg_results_surface, index=priors_name_lst,
                                                  columns=structure_lst)
    patient_reg_results_dice_df = pd.DataFrame(patient_reg_results_dice, index=priors_name_lst, columns=structure_lst)

    return patient_reg_results_cm_df, patient_reg_results_surface_df, patient_reg_results_dice_df


def check_priors_cm_dist(context, cur_similarity_list, resolution=0.5, prior_centering='mean', cache=None):

    # A client can provide a cache across multiple calls to avoid recomputing values for the same priors.
    # The client should not modify the contents of the cache nor depend on its content. If no cache was
    # provided, create an empty one to make the logic simpler below. However, the contents of the cache
    # will then not be available to subsequent calls.
    if cache is None:
        cache = {}

    structure_lst = list(itertools.product(BGStructure.unmerged_choices(), Side.choices))
    structure_name_lst = ['{}_{}'.format(bg_structure.name, side.name[:1]) for bg_structure, side in structure_lst]
    structure_count = len(structure_lst)

    good_priors_name_lst = []
    compute_priors_name_lst = []
    compute_priors_files_lst = []
    for prior_name in cur_similarity_list:
        if prior_name in cache:
            sis_log(INFO, SLM.PREDICTION, None, "Cached center of mass measurements available for %s", prior_name)
            good_priors_name_lst.append(prior_name)
            continue

        priors_files = []
        for bg_structure, side in structure_lst:
            pf = context.get_files_path(DSN.PRIOR_REGTO_BET_3D(prior_name, modality=T2_MODALITY, side=side,
                                                               bg_structure=bg_structure, thr05=True, cropped=False))
            if not os.path.exists(pf):
                sis_log(WARNING, SLM.PREDICTION, SLC.WORKFILES_NOT_FOUND, "Missing prior file: %s", pf)
            else:
                priors_files.append(pf)

        if len(priors_files) < structure_count:
            sis_log(WARNING, SLM.PREDICTION, SLC.WORKFILES_NOT_FOUND,
                    "Skipping center of mass measurements for %s due to missing files", prior_name)

        sis_log(INFO, SLM.PREDICTION, None, "Computing center of mass measurements for %s", prior_name)
        good_priors_name_lst.append(prior_name)
        compute_priors_name_lst.append(prior_name)
        compute_priors_files_lst.append(priors_files)

    for compute_prior_name, compute_result in zip(compute_priors_name_lst,
                                                  compute_priors_cm_dist(compute_priors_files_lst)):
        cache[compute_prior_name] = compute_result

    priors_cm_list = np.array([cache[prior] for prior in good_priors_name_lst])

    str_cm_avg = compute_structure_centers(priors_cm_list, prior_centering)
    priors_cm_diff = priors_cm_list - str_cm_avg

    priors_cm_dist = np.zeros([len(good_priors_name_lst), structure_count])
    for i in range(len(good_priors_name_lst)):
        for j in range(structure_count):
            priors_cm_dist[i][j] = np.linalg.norm(priors_cm_diff[i][j]) * resolution

    priors_cm_dist_df = pd.DataFrame(priors_cm_dist, index=good_priors_name_lst, columns=structure_name_lst)

    return priors_cm_dist_df


def center16(a, axis):
    if axis != 0:
        raise ValueError('Expected axis=0')

    mean_a = np.mean(a, axis=0)
    while len(a) > 16:
        norm_a = np.linalg.norm(a - mean_a, axis=1)
        worst_i = np.argmax(norm_a)

        # Rather than recalculating the mean from scratch, we *could* just shift the
        # old mean by 1/N of the point just removed. I think.
        #    mean_a = mean_a - a[worst_i] / len(a)
        #    a = np.delete(a, worst_i, axis=0)

        a = np.delete(a, worst_i, axis=0)
        mean_a = np.mean(a, axis=0)

    return mean_a


def compute_structure_centers(priors_cm_list, prior_centering='mean'):
    """

    :param priors_cm_list: N priors * M structures * D dimensions (typically 16, 6 and 3)
    :type priors_cm_list: numpy.array
    :param prior_centering: method to compute prior centers ('mean' or 'median')
    :return: structure center coordinates - M structures * D dimensions
    """
    n_priors, n_structures, n_dim = priors_cm_list.shape

    result = np.zeros(priors_cm_list[0].shape)
    method = {'mean': np.mean, 'median': np.median, 'center16': center16}[prior_centering]

    for i in range(n_structures):
        result[i] = method(priors_cm_list[:, i, :], axis=0)

    return result


# def compute_priors_cm_dist(priors_files_lst):
#     """
#
#     :param priors_files_lst: a list of file lists
#     :return: a list of center-of-mass lists
#     """
#     return spc.call('compute_priors_cm_dist.py', priors_files_lst)
#
#
# def get_most_similar_prior(context):
#     similarity_file = context.get_files_path(DSN.PREDICTION_SIMILARITY_SUMMARY)
#     with open(similarity_file, 'r') as sim_file:
#         reader = csv.reader(sim_file)
#         similar_brain_lst = list(reader)
#     most_similar_brain = similar_brain_lst[0][0]
#     return most_similar_brain
#
#
# def compute_priors_cm_metrics(priors_cm_dist_df):
#     priors_cm_distr = priors_cm_dist_df.mean(axis=1) + 2 * priors_cm_dist_df.std(axis=1)
#     return {key: value for key, value in zip(priors_cm_distr.index, priors_cm_distr)}
#
#
# # noinspection PyUnresolvedReferences
# def detect_failed_registrations(patient_reg_results_cm_df, patient_reg_results_surface_df):
#
#     """
#     Detect if a a prior have registration error of more than 3mm compare to the most simila brain
#
#     :param patient_reg_results_cm_df: Data frame that contain the priors center of mass distances
#     :param patient_reg_results_surface_df: Data frame that contain the priors average surface distances
#     :return: list of brains that faild the registration
#
#     """
#     RESOLUTION = 0.5  # resolution of the image (for calculating real distance)
#     DISTANCE_THRESHOLD = 3  # distance in mm that above it the registration is considered failed
#
#     priors_above_threshold_cm = (RESOLUTION * patient_reg_results_cm_df.mean(axis=1)) > DISTANCE_THRESHOLD
#     priors_above_threshold_surf = (RESOLUTION * patient_reg_results_surface_df.mean(axis=1)) > DISTANCE_THRESHOLD
#     failed_reg_priors_cm = priors_above_threshold_cm[priors_above_threshold_cm == True].index.tolist()
#     failed_reg_priors_surf = priors_above_threshold_surf[priors_above_threshold_surf == True].index.tolist()
#     failed_reg_priors = failed_reg_priors_cm + list(set(failed_reg_priors_surf) - set(failed_reg_priors_cm))
#
#     return failed_reg_priors
#
#
# def check_patient_reg(context, most_similar_prior, priors_list):
#
#     patient_reg_results_cm_df, patient_reg_results_surface_df, patient_reg_results_dice_df = \
#         check_reg_results(context, most_similar_prior, priors_list)
#
#     # list of failed registered brain
#     failed_reg_priors_name, failed_reg_priors_dist_values = \
#         detect_failed_registrations(patient_reg_results_cm_df, patient_reg_results_surface_df)
#
#     return failed_reg_priors_name, failed_reg_priors_dist_values, patient_reg_results_cm_df, \
#         patient_reg_results_surface_df

# Test the codes
if __name__ == '__main__':
    #label, _ = _load('/home/naxos2-raid18/orens/Projects/Seg_GPiGPe/AttentionUnet/checkpoints/experiment_unet_ct_dsv_mri_gpe94/mult_prj_94/2_lbl.nii.gz')
    #pred, header = _load('/home/naxos2-raid18/orens/Projects/Seg_GPiGPe/AttentionUnet/checkpoints/experiment_unet_ct_dsv_mri_gpe94/mult_prj_94/2_pred.nii.gz')

    label, _ = _load('/home/naxos2-raid18/orens/DBS_for_orens/STN_project/STN_checkpoints/experiment_unet_stn_sn_rn_r43/mult_prj_70/197_lbl.nii.gz')
    pred, header = _load('/home/naxos2-raid18/orens/DBS_for_orens/STN_project/STN_checkpoints/experiment_unet_stn_sn_rn_r43/mult_prj_70/197_pred.nii.gz')

    # Test new version of remove_islands
    cleaned_pred = remove_islands(pred)
    nib.Nifti1Image(cleaned_pred, affine=None, header=None).to_filename('/home/naxos2-raid18/orens/DBS_for_orens/STN_project/STN_checkpoints/experiment_unet_stn_sn_rn_r43/mult_prj_70/197_test.nii.gz')

    # Remove unwanted "islands" (post processing)
    cleaned_pred = remove_islands_gp(pred)
    # nib.Nifti1Image(cleaned_pred, affine=None, header=header).to_filename(
    #     os.path.join('/home/naxos2-raid18/orens/Projects/Seg_GPiGPe/AttentionUnet/checkpoints/experiment_unet_ct_dsv_mri_gpe94/mult_prj', "cleaned_pred_2.nii.gz"))

    # Calculate volumes
    volumes = calculate_volume(label, pixdim=[0.391])
    print("Label measured volumes [cm^3]:")
    print("GPe: right {}, left {}".format(volumes[0] / 1000, volumes[1] / 1000))
    print("GPi: right {}, left {}".format(volumes[2] / 1000, volumes[3] / 1000))
    print("")

    volumes = calculate_volume(cleaned_pred, pixdim=[0.391])
    print("Segmentation measured volumes [cm^3]:")
    print("GPe: right {}, left {}".format(volumes[0] / 1000, volumes[1] / 1000))
    print("GPi: right {}, left {}".format(volumes[2] / 1000, volumes[3] / 1000))
    print("")

    # Measure center of mass distance
    cm_dist, cm_diff = measure_cm_dist_wrapper(cleaned_pred, label, pixdim=[0.391]) # Still need to work on this
    print("Center of mass difference [mm]:")
    print("GPe: right {}, left {}".format(cm_dist[0, 0], cm_dist[0, 1]))
    print("GPi: right {}, left {}".format(cm_dist[1, 0], cm_dist[1, 1]))

    a = 1
    #print("Center of mass distance [pixels] = {}".format(measure_CM_dist(label, cleaned_pred)))
    #print("Center of mass distance [mm] = {}".format(measure_CM_dist_real_size(label, cleaned_pred, pixdim=[0.391])))





