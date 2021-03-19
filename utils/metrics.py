# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import cv2
from tqdm import tqdm

import copy
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue
from skimage import measure
from scipy import ndimage
from utils.measure_registration_results import measure_CM_dist
from scipy.spatial.distance import directed_hausdorff

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {'overall_acc': acc,
            'mean_acc': acc_cls,
            'freq_w_acc': fwavacc,
            'mean_iou': mean_iu}


def dice_score_list(label_gt, label_pred, n_class):
    """

    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)


def dice_score(label_gt, label_pred, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """

    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    #print(n_class)
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32).flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores

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

def dice_score_average_left_right(label_gt, label_pred, n_class=2, connectivity=26):
    """
    This function is intended to be used when for each segmentation class there are two sides (for brain imaging
    we have some symmetry between the left and right side). We output the DICE score for each side, as well as the
    averaged DICE - which is typically what we want

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    import cc3d  # pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)
    from copy import deepcopy

    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_gt.shape)
    dice_scores_av = np.zeros(n_class, dtype=np.float32)
    dice_scores_two_sides = np.zeros((n_class - 1, 2), dtype=np.float32) # Number of classes X left + right
    #print(n_class)
    for class_id in range(n_class):
        # Inputs per class
        img_A = np.array(label_gt == class_id, dtype=np.uint16)#.flatten()
        img_B = np.array(label_pred == class_id, dtype=np.uint16)#.flatten()

        if class_id != 0: # 0 is background
            # Divide into left and right using connected components
            img_A_cc = cc3d.connected_components(img_A, connectivity=connectivity)  # 26-connected (default)
            img_B_cc = cc3d.connected_components(img_B, connectivity=connectivity)  # 26-connected (default)

            # I found a bug in cc3d for img_B_cc - patient PD092
            n_class_for_two_sides = 3 # Background, left and right parts (0, 1, 2)
            if np.max(img_B_cc) > n_class_for_two_sides - 1:
                print('Found outlier value in cc3d, correcting...')
                img_B_cc[img_B_cc == np.max(img_B_cc)] = n_class_for_two_sides - 1

            # Make sure the CC are in the same order
            img_A_cc = check_cc_of_same_object(img_A_cc, img_B_cc, class_id)

            # Extract individual components - img A
            for segid in range(1, np.max(img_A_cc) + 1):
                extracted_image_A = img_A_cc * (img_A_cc == segid)
                extracted_image_A[extracted_image_A != 0] = 1
                extracted_image_A = extracted_image_A.astype(np.float32).flatten()

                extracted_image_B = img_B_cc * (img_B_cc == segid)
                extracted_image_B[extracted_image_B != 0] = 1
                extracted_image_B = extracted_image_B.astype(np.float32).flatten()

                score = 2.0 * np.sum(extracted_image_A * extracted_image_B) / (np.sum(extracted_image_A) + np.sum(extracted_image_B) + epsilon)
                #print(score)
                # Each row is a different structure, columns are left and right
                dice_scores_two_sides[class_id - 1, segid - 1] = score
        else:
            img_A = img_A.flatten()
            img_B = img_B.flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores_av[class_id] = score

    # Averaged DICE on both sides
    dice_scores_av[1:n_class] = np.mean(dice_scores_two_sides, axis=1)
    return dice_scores_av, dice_scores_two_sides


def precision_and_recall(label_gt, label_pred, n_class):
    from sklearn.metrics import precision_score, recall_score
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall

def distance_metric(seg_A, seg_B, dx, k):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """

    # Extract the label k from the segmentation maps to generate binary maps
    seg_A = (seg_A == k)
    seg_B = (seg_B == k)

    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            contours, hierarchy = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, hierarchy = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd

def distance_metric_new(seg_A, seg_B, vox_size=1):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.

        seg_A - Network segmentation
        seg_B - Ground truth
        vox_size - Voxel size [mm] (scalar). Assumes isotropic voxel size
        """
    import cc3d  # pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)
    print("Number of cpu : ", multiprocessing.cpu_count())

    # Number of classes
    n_class = int(np.max(seg_B.ravel()))

    # Dimensions
    X, Y, Z = seg_A.shape

    hd = np.zeros(n_class) # Hausdorff distance per class
    msd = np.zeros(n_class) # Mean surface distance per class
    for k in range(n_class):
        # Extract the label k from the segmentation maps to generate binary maps
        seg_A_tmp = copy.deepcopy(seg_A)
        seg_B_tmp = copy.deepcopy(seg_B)

        # Exclude the background (0)
        seg_A_tmp[seg_A != (k + 1)] = 0
        seg_B_tmp[seg_B != (k + 1)] = 0
        seg_A_tmp[seg_A_tmp != 0] = 1
        seg_B_tmp[seg_B_tmp != 0] = 1

        # Calculate the Hausdorff distance per each slice, only if both slices contain information
        tmp_hd = 0
        first_time_flag = 1

        # Get all contour voxels for the 3D objects
        print("Extracting contours, k = {}".format(k))
        for z in range(Z):
            # Binary mask at this slice
            slice_A = seg_A_tmp[:, :, z].astype(np.uint8)
            slice_B = seg_B_tmp[:, :, z].astype(np.uint8)

            # Create a list of indices of non-zero pixels
            if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
                # Get the contours of the slices
                edge_img_A = find_edges_seg(slice_A)
                edge_img_B = find_edges_seg(slice_B)

                # The distance is defined only when both contours exist on this slice
                tmp1 = np.array(np.where(edge_img_A != 0))
                tmp1_1 = z * np.ones(tmp1.shape[1])  # Add the slice dimension
                tmp2 = np.array(np.where(edge_img_B != 0))
                tmp2_1 = z * np.ones(tmp2.shape[1])  # Add the slice dimension
                if first_time_flag == 1:
                    qA = np.append(tmp1, tmp1_1.reshape(1, tmp1.shape[1]), axis=0).transpose()  # List of XYZ coordinates
                    qB = np.append(tmp2, tmp2_1.reshape(1, tmp2.shape[1]), axis=0).transpose()  # List of XYZ coordinates
                    first_time_flag = 0
                else:
                    q_tmp = np.append(tmp1, tmp1_1.reshape(1, tmp1.shape[1]), axis=0).transpose()
                    p_tmp = np.append(tmp2, tmp2_1.reshape(1, tmp2.shape[1]), axis=0).transpose()
                    qA = np.append(qA, q_tmp, axis=0)
                    qB = np.append(qB, p_tmp, axis=0)

        # Rescale points according to voxel size (for now voxel is assumed to be isotropic) [mm]
        qA = qA * vox_size
        qB = qB * vox_size

        # Mean surface distance
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("Calculating distance matrix")
        # Distance matrix between point sets

        # Serial calculation
        # ####################################
        # M = np.zeros((len(qA), len(qB)))
        # for i in tqdm(range(len(qA))):
        #     for j in range(len(qB)):
        #         M[i, j] = np.linalg.norm(qA[i, :] - qB[j, :])
        # ####################################

       # Compute the mean surface distance in parallel
        M = mp_run(qA, qB)

        msd[k] = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1)))
        hd[k] = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))])
    return hd, msd

def distance_metric_new_left_right(seg_A, seg_B, vox_size=1):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.

        seg_A - Network segmentation
        seg_B - Ground truth
        vox_size - Voxel size [mm] (scalar). Assumes isotropic voxel size
        """
    import cc3d  # pip install connected-components-3d --no-binary :all: (https://pypi.org/project/connected-components-3d/)
    print("Number of cpu : ", multiprocessing.cpu_count())

    # Number of classes
    n_class = int(np.max(seg_B.ravel()))

    # Dimensions
    X, Y, Z = seg_A.shape

    hd = np.zeros((n_class, 2)) # Hausdorff distance per class, left and right
    msd = np.zeros((n_class, 2)) # Mean surface distance per class, left and right
    for k in range(n_class):
        # Extract the label k from the segmentation maps to generate binary maps
        seg_A_tmp = copy.deepcopy(seg_A.astype(np.uint8))
        seg_B_tmp = copy.deepcopy(seg_B.astype(np.uint8))

        # Exclude the background (0)
        seg_A_tmp[seg_A != (k + 1)] = 0
        seg_B_tmp[seg_B != (k + 1)] = 0
        seg_A_tmp[seg_A_tmp != 0] = 1
        seg_B_tmp[seg_B_tmp != 0] = 1

        # Divide into left and right using connected components
        img_A_cc = cc3d.connected_components(seg_A_tmp)  # 26-connected (default)
        img_B_cc = cc3d.connected_components(seg_B_tmp)  # 26-connected (default)

        # I found a bug in cc3d for img_B_cc - patient PD092
        n_class_for_two_sides = 3  # Background, left and right parts (0, 1, 2)
        if np.max(img_B_cc) > n_class_for_two_sides - 1:
            print('Found outlier value in cc3d, correcting...')
            img_B_cc[img_B_cc == np.max(img_B_cc)] = n_class_for_two_sides - 1

        # Make sure the CC are in the same order
        img_A_cc = check_cc_of_same_object(img_A_cc, img_B_cc, class_id=k)
        num_cc_max = np.max(img_A_cc)

        tmp_hd = np.zeros(num_cc_max)
        tmp_msd = np.zeros(num_cc_max)
        for cc in range(num_cc_max):
            # Calculate the Hausdorff distance per each slice, only if both slices contain information
            #tmp_hd = 0
            first_time_flag = 1

            img_A_cc_tmp = copy.deepcopy(img_A_cc)
            img_B_cc_tmp = copy.deepcopy(img_B_cc)

            img_A_cc_tmp[img_A_cc_tmp != cc + 1] = 0
            img_B_cc_tmp[img_B_cc_tmp != cc + 1] = 0
            img_A_cc_tmp[img_A_cc_tmp > 0] = 1
            img_B_cc_tmp[img_B_cc_tmp > 0] = 1

            # Get all contour voxels for the 3D objects
            print("Extracting contours, k = {}".format(k))
            qA = np.array([-1])
            qB = np.array([-1])
            for z in range(Z):
                # Binary mask at this slice
                slice_A = img_A_cc_tmp[:, :, z].astype(np.uint8)
                slice_B = img_B_cc_tmp[:, :, z].astype(np.uint8)

                # Create a list of indices of non-zero pixels
                if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
                    # Get the contours of the slices
                    edge_img_A = find_edges_seg(slice_A)
                    edge_img_B = find_edges_seg(slice_B)

                    # The distance is defined only when both contours exist on this slice
                    tmp1 = np.array(np.where(edge_img_A != 0))
                    tmp1_1 = z * np.ones(tmp1.shape[1])  # Add the slice dimension
                    tmp2 = np.array(np.where(edge_img_B != 0))
                    tmp2_1 = z * np.ones(tmp2.shape[1])  # Add the slice dimension
                    if first_time_flag == 1:
                        qA = np.append(tmp1, tmp1_1.reshape(1, tmp1.shape[1]), axis=0).transpose()  # List of XYZ coordinates
                        qB = np.append(tmp2, tmp2_1.reshape(1, tmp2.shape[1]), axis=0).transpose()  # List of XYZ coordinates
                        first_time_flag = 0
                    else:
                        q_tmp = np.append(tmp1, tmp1_1.reshape(1, tmp1.shape[1]), axis=0).transpose()
                        p_tmp = np.append(tmp2, tmp2_1.reshape(1, tmp2.shape[1]), axis=0).transpose()
                        qA = np.append(qA, q_tmp, axis=0)
                        qB = np.append(qB, p_tmp, axis=0)

            if len(qA) != 1 and len(qA) != 1:
                # Rescale points according to voxel size (for now voxel is assumed to be isotropic) [mm]
                qA = qA * vox_size
                qB = qB * vox_size

                # Mean surface distance
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # qA has a valid number
                print("Calculating distance matrix")
                # Distance matrix between point sets

                # Serial calculation
                # ####################################
                # M = np.zeros((len(qA), len(qB)))
                # for i in tqdm(range(len(qA))):
                #     for j in range(len(qB)):
                #         M[i, j] = np.linalg.norm(qA[i, :] - qB[j, :])
                # ####################################

                # Compute the mean surface distance in parallel
                M = mp_run(qA, qB)

                tmp_msd[cc] = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1)))
                tmp_hd[cc] = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))])
            else:
                # Can't compute MSD and HD
                tmp_msd[cc] = -1.0
                tmp_hd[cc] = -1.0

        # Each row is for a different organ, each column is left / right
        try:
            msd[k] = tmp_msd
        except:
            print("Failed to calculate MSD")
        try:
            hd[k] = tmp_hd
        except:
            print("Failed to calculate HD")

    return hd, msd

class split_data_class:
    def __init__(self):
        self.qA = 0
        self.qB = 0
        self.output=0

    def place(self, qA, qB):
        self.qA = qA
        self.qB = qB

# Queue for multi-processing with multiple CPUs
def mp_worker(queue, f, *argv):
    queue.put(f(*argv))

def mp_run(qA, qB):
    n_threads = multiprocessing.cpu_count()

    processes = [None] * n_threads
    #results = [None] * n_threads
    queue = Queue()

    # Split the data
    qA_split = np.array_split(qA, n_threads, axis=0)

    # Init and start processes
    for t in range(n_threads):
        qA_chunck = qA_split[t]
        p = Process(target=mp_worker, args=(queue, calc_dist_mat, qA_chunck, qB))
        processes[t] = p
        p.start()

    # Collect process output from the queue
    for t in tqdm(range(n_threads)):
        #results[t] = queue.get()
        if t == 0:
            results = queue.get()
        else:
            results = np.concatenate((results, queue.get()), axis=0)

    # Wait for the processes to finish
    for p in processes:
        p.join()

    return results

def calc_dist_mat(qA, qB):
    # qA = data.qA
    # qB = data.qB
    M = np.zeros((len(qA), len(qB)))
    for i in range(len(qA)):
        for j in range(len(qB)):
            M[i, j] = np.linalg.norm(qA[i, :] - qB[j, :])
    return M

def find_edges_seg(img):
    '''
    Find the edges of a segmentation image, composed on 0's and 1's
    Not numerically efficient, but accurate

    :param img: Numpy array represanting the segmentation image
    :return: edges image of the input
    '''
    dims = img.shape
    edges = np.zeros(dims)

    for ii in range(dims[0]):
        for jj in range(dims[1]):
            if img[ii, jj] != 0:
                cube_sum = np.sum(img[ii - 1:ii + 2, jj - 1:jj + 2])
                if cube_sum > 0 and cube_sum < 9:  # 8 connected + center
                    edges[ii, jj] = 1

    return edges

# Test
if __name__ == '__main__':
    import nibabel as nib
    test_case = 3
    if test_case == 1:
        seg_A_name = '/home/naxos2-raid18/orens/Projects/Seg_GPiGPe/AttentionUnet/checkpoints/experiment_unet_ct_dsv_mri_gpe94/mult_prj_94/2_pred.nii.gz'
        seg_B_name = '/home/naxos2-raid18/orens/Projects/Seg_GPiGPe/AttentionUnet/checkpoints/experiment_unet_ct_dsv_mri_gpe94/mult_prj_94/2_lbl.nii.gz'
        n_class = 3
        connectivity = 6
    elif test_case == 2:
        seg_A_name = '/home/naxos2-raid18/orens/DBS_for_orens/Thalamus_project/Thalamus_checkpoints/experiment_unet_thalamus_r10/mult_prj/23_pred.nii.gz'
        seg_B_name = '/home/naxos2-raid18/orens/DBS_for_orens/Thalamus_project/Thalamus_checkpoints/experiment_unet_thalamus_r10/mult_prj/23_lbl.nii.gz'
        n_class = 2
        connectivity = 6
    elif test_case == 3:
        seg_B_name = '/home/naxos2-raid18/orens/DBS_for_orens/3Tseg_project/checkpoints/experiment_unet_r7/mult_prj/0_pred.nii.gz'
        seg_A_name = '/home/naxos2-raid18/orens/DBS_for_orens/3Tseg_project/checkpoints/experiment_unet_r7/mult_prj/0_lbl.nii.gz'
        n_class = 6
        connectivity = 26

    seg_A = nib.load(seg_A_name).get_fdata()
    seg_B = nib.load(seg_B_name).get_fdata()

    av_dice, dices = dice_score_average_left_right(seg_A, seg_B, n_class=n_class, connectivity=connectivity)
    #dice_2_sides = dice_score(seg_A, seg_B, n_class=3)
    print("Averaged DICE = {}".format(av_dice))
    #print("DICE two sides = {}".format(dice_2_sides))

    # slice 45
    hd, msd = distance_metric_new_left_right(seg_A, seg_B, vox_size=0.390625)
    print("Hausdorff distance [mm] = {}".format(hd))
    print("Mean surface distance [mm] = {}".format(msd))

    # Original code
    #mean_md2, mean_hd2 = distance_metric(seg_A, seg_B, dx=2, k=1)
    #print(mean_hd2)