'''
This script copies the T@ / SWI coronal files and corresponding manual segmentations into the correct locations.
'''
import os, sys
import re
import shutil
import numpy as np
import random

#                       Functions
# ----------------------------------------------------

def list_swi_patients(source_folder, patients):
    swi_list = []
    for patient in patients:
        path_dir = os.path.join(source_folder, patient)
        lst = [f for f in os.listdir(path_dir) if not f.startswith('.') and f.lower().find('swi') != -1]
        if lst:
            swi_list.append(patient)

    return swi_list, len(swi_list)


def copy_patients(source_dir, dest_dir, patients, type_of_dirs):
    kk = 1
    for patient in patients:
        print('Patient {} - {}/{}'.format(patient, kk, len(patients)))

        # Create patient directory
        if not os.path.exists(os.path.join(dest_dir, 'Patients', patient)):
            os.mkdir(os.path.join(dest_dir, 'Patients', patient))
            print("Created directory for patient {}".format(patient))

        # Create segmentation folders and T2, SWI folders
        for type_dir in type_of_dirs:
            if not os.path.exists(os.path.join(dest_dir, 'Patients', patient, type_dir)):
                os.mkdir(os.path.join(dest_dir, 'Patients', patient, type_dir))
                print('{}: Created directory {}'.format(patient, type_dir))
                if type_dir == '7T_SWI' or type_dir == '7T_T2':
                    if not os.path.exists(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Coronal')):
                        os.mkdir(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Coronal'))
                        print('{}: Created sub-directory {}'.format(patient, type_dir + '/Coronal'))
                    if not os.path.exists(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Axial')):
                        os.mkdir(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Axial'))
                        print('{}: Created sub-directory {}'.format(patient, type_dir + '/Axial'))
            elif not os.path.exists(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Coronal')):
                if type_dir == '7T_SWI' or type_dir == '7T_T2':
                    os.mkdir(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Coronal'))
                    print('{}: Created sub-directory {}'.format(patient, type_dir + '/Coronal'))

        # Copy files if they exist
        for type_dir in type_of_dirs:
            src_folder = os.path.join(source_dir, patient)
            dest_folder = os.path.join(dest_dir, 'Patients', patient, type_dir)
            try:
                if type_dir == '7T_SWI' or type_dir == '7T_T2':
                    type_dir = type_dir + '_cor'
                    # Scans
                    [shutil.copyfile(os.path.join(src_folder, f), os.path.join(dest_folder, 'Coronal', f)) for f in
                     os.listdir(src_folder) if f.find(type_dir) != -1 and f.lower().find('regto') == -1
                     and f.find('STN') == -1 and f.find('SN') == -1 and f.find('RN') == -1 and f.find('axi') == -1]
                else:
                    # Segmentations
                    [shutil.copyfile(os.path.join(src_folder, f), os.path.join(dest_folder, f)) for f in
                     os.listdir(src_folder) if f.find(type_dir) != -1]

                print('{}: Copied {} files to {}'.format(patient, type_dir, dest_folder))
            except:
                print('{}: No files were copied'.format(patient))


        kk += 1

    return 0


# Parameters
# ----------------------------------------------------
source_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/STN_Dataset_Reviewed'

# Types of files
type_of_dirs = ['STN', 'SN', 'RN', '7T_T2']
type_of_patho = ['C', 'SLEEP', 'ET', 'P', 'DYS'] # SLEEP are also control

# List of patient folders
source_patients = sorted([f for f in os.listdir(source_dir) if not f.startswith('.') and f.find('C062') == -1
                                                                                     and f.find('ET039') == -1
                                                                                     and f.find('P046') == -1
                                                                                     and f.find('PD048') == -1
                                                                                     and f.find('PD053') == -1
                                                                                     and f.find('P042') == -1])
src_len = len(source_patients)

# Types of scans per pathology type
patho_lens = []
patho_patients = []
for ii in range(len(type_of_patho)):
    tmp_list = (sorted([f for f in source_patients if f.find(type_of_patho[ii]) != -1]))
    if type_of_patho[ii] == 'P':
        tmp_list_2 = (sorted([f for f in tmp_list if f.find('SLEEP') == -1]))
        patho_patients.append(tmp_list_2)
    else:
        patho_patients.append(tmp_list)
    patho_lens.append(len(patho_patients[ii]))

print("Type of pathology: {}".format(type_of_patho))
print("Number of subjects per pathology: {}".format(patho_lens))

# List of SWI patients
swi_list, swi_list_len = list_swi_patients(source_dir, source_patients)
print("{} SWI patients: {}".format(swi_list_len, swi_list))

# Perform subjects division - according to the division in type_of_patho
balance_case = '70_30' #'balanced'
if balance_case == 'balanced':
    train_num_patients = [5, 28, 10, 23, 0] # ['C', 'SLEEP', 'ET', 'P', 'DYS']
    test_num_patients = [5, 28, 10, 24, 2] # ['C', 'SLEEP', 'ET', 'P', 'DYS']
elif balance_case == '70_30': # 70% for training
    train_num_patients = [8, 40, 15, 33, 0]  # ['C', 'SLEEP', 'ET', 'P', 'DYS']
    test_num_patients = [2, 16, 5, 14, 2]  # ['C', 'SLEEP', 'ET', 'P', 'DYS']

# Set random seed (original I used is 42)
random.seed(42)

# Start by working on each pathology type
num_of_pathologies = len(type_of_patho)
for ii in range(num_of_pathologies):
    # current list of patients - exclude SWI patients
    curr_list = [f for f in patho_patients[ii] if not (f in swi_list)]

    # Choose randomly the patients for training and test
    training_patients = random.sample(curr_list, train_num_patients[ii])
    test_patients = [f for f in curr_list if not (f in training_patients)]

    # Training
    # -----------------------
    if balance_case == 'balanced':
        dest_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/STN_project/T2_coronal_train_verified'
    elif balance_case == '70_30':
        dest_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/STN_project/T2_coronal_train_verified_70_30'
    copy_patients(source_dir, dest_dir, training_patients, type_of_dirs)

    # Test
    # -----------------------
    if balance_case == 'balanced':
        dest_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/STN_project/T2_coronal_test_verified'
    elif balance_case == '70_30':
        dest_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/STN_project/T2_coronal_test_verified_70_30'
    copy_patients(source_dir, dest_dir, test_patients, type_of_dirs)

print('Done.')




