'''
This script arranges the data for 3T scans segmentations.
Some of the scans in 3T_data are actually 1.5T, so we won't use them at the moment

STN
SN
RN
GPe
GPi

'''
import os, sys
import numpy as np
import shutil

# -- Parameters --
src_folder = '/home/naxos2-raid18/orens/DBS_for_orens/3Tseg_project/3T_Data'
patients_folder = '/home/naxos2-raid18/orens/DBS_for_orens/3Tseg_project/Patients'
use_1p5_flag = True
folders_list = ['3T_T2', 'STN', 'SN', 'RN', 'GP_Labels']

# stn_seg_folders = ['/home/naxos2-raid18/orens/DBS_for_orens/STN_project/T2_coronal_test_verified_70_30/Patients',
#                    '/home/naxos2-raid18/orens/DBS_for_orens/STN_project/T2_coronal_train_verified_70_30/Patients']

stn_seg_folders = ['/home/naxos2-raid18/orens/DBS_for_orens/STN_Dataset_Reviewed']

gp_seg_folders = ['/home/naxos2-raid18/orens/DBS_for_orens/GPi_project/FINAL/Part1/Patients',
                  '/home/naxos2-raid18/orens/DBS_for_orens/GPi_project/FINAL/Part2/Patients',
                  '/home/naxos2-raid18/orens/DBS_for_orens/GPi_project/FINAL/Part3/Patients']

# -- Process --
if not os.path.exists(patients_folder):
    os.mkdir(patients_folder)

# Get list of patients
patients = sorted([f for f in os.listdir(src_folder) if not f.startswith('.')])

# For each patient
for pid, patient in enumerate(patients):
    print(' ')
    print('             Working on {} - {}/{}'.format(patient, pid + 1, len(patients)))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # Create directories
    if not os.path.exists(os.path.join(patients_folder, patient)):
        os.mkdir(os.path.join(patients_folder, patient))

    # Check if patient has 3T scan or not - if so, continue
    if not use_1p5_flag:
        chk = [f for f in os.listdir(os.path.join(src_folder, patient)) if not f.startswith('.') and f.find('3T') != -1]
        if not chk:
            print("Patient has no 3T scan, skipping...")
            continue

    for fold in folders_list:
        print(' ')
        print("Working on folder {}".format(fold))
        # Create directories
        dest_folder = os.path.join(patients_folder, patient, fold)
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
            if not os.path.exists(os.path.join(dest_folder, 'Axial')) and fold == '3T_T2':
                os.mkdir(os.path.join(dest_folder, 'Axial'))

        # Find scans and segmentations
        if fold == '3T_T2':
            # 3T / 1.5T images
            dest_folder_ax = os.path.join(dest_folder, 'Axial') # Keep in mind that some of the data might be 1.5T...
            src_pat_folder = os.path.join(src_folder, patient, [f for f in os.listdir(os.path.join(src_folder, patient)) if not f.startswith('.')][0])
            files = [f for f in os.listdir(src_pat_folder) if not f.startswith('.')]
            for file in files:
                shutil.copyfile(os.path.join(src_pat_folder, file), os.path.join(dest_folder_ax, file))
                print("Copied file {} to destination {}".format(file, dest_folder_ax))
        else:
            # Look for segmentations
            if fold.lower() == 'stn' or fold.lower() == 'sn' or fold.lower() == 'rn':
                # STN / SN / RN
                seg_search_folders = stn_seg_folders
            else:
                # GP
                seg_search_folders = gp_seg_folders

            # Search for the segmentations
            for search_folder in seg_search_folders:
                if fold != 'GP_Labels':
                    curr_search_folder = os.path.join(search_folder, patient)
                else:
                    curr_search_folder = os.path.join(search_folder, patient, fold)
                try:
                    files_to_copy = [f for f in os.listdir(curr_search_folder) if not f.startswith('.')]
                    if fold != 'GP_Labels':
                        files_to_copy = [f for f in files_to_copy if f.find(fold) != -1]
                except:
                    print("{} not in {}, or empty folder for segmentation {}".format(patient, curr_search_folder, fold))
                    continue

                for file in files_to_copy:
                    shutil.copyfile(os.path.join(curr_search_folder, file), os.path.join(dest_folder, file))
                    print("Copied file {} to destination {}".format(file, dest_folder))






print('- done -')
















