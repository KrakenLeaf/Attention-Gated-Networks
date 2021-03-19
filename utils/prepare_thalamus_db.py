'''
Prepare the Thalamus segmentation db, The data itself sits in Jinyoung's directory
'''
import os, sys
import numpy as np
import shutil


source_db = '/home/naxos2-raid18/orens/Jinyoung_link/datasets/thalamus'
target_db = '/home/naxos2-raid18/orens/DBS_for_orens/Thalamus_project/Patients'
dirs = ['7T_T1', '7T_DTI_B0', '7T_DTI_FA', 'Thalamus']

patients = sorted([f for f in os.listdir(source_db) if not f.startswith('.')])

for patient in patients:
    print('Patient: {}'.format(patient))
    print('----------------------')

    # Create patient directory
    pat_dir = os.path.join(target_db, patient)
    if not os.path.exists(pat_dir):
        os.makedirs(pat_dir)

    # Create sub-directories and copy files
    for dir in dirs:
        # Make directories
        if not os.path.exists(os.path.join(pat_dir, dir)):
            os.makedirs(os.path.join(pat_dir, dir))

        if dir != 'Thalamus':
            if not os.path.exists(os.path.join(pat_dir, dir, 'Axial')):
                os.makedirs(os.path.join(pat_dir, dir, 'Axial'))

        # Copy file
        if dir != 'Thalamus':
            src_dir = os.path.join(source_db, patient, 'images')
        else:
            src_dir = os.path.join(source_db, patient, 'gt')

        if dir == '7T_T1':
            src_files = [f for f in os.listdir(src_dir) if f.find('T1') != -1]
        elif dir == '7T_DTI_B0':
            src_files = [f for f in os.listdir(src_dir) if f.find('B0') != -1 and f.find('register') != -1]
        elif dir == '7T_DTI_FA':
            src_files = [f for f in os.listdir(src_dir) if f.find('FA') != -1 and f.find('register') != -1]
        elif dir == 'Thalamus':
            src_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]

        for src_file in src_files:
            print('{}: copied {}'.format(patient, src_file))
            if dir != 'Thalamus':
                shutil.copyfile(os.path.join(src_dir, src_file), os.path.join(pat_dir, dir, 'Axial', src_file))
            else:
                shutil.copyfile(os.path.join(src_dir, src_file), os.path.join(pat_dir, dir, src_file))


print(' - done -')









