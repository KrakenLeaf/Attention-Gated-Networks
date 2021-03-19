'''
This script copies the T@ / SWI coronal files and corresponding manual segmentations into the correct locations.
'''
import os, sys
import re
import shutil

# Parameters
# ----------------------------------------------------
case = 2
if case == 1:
# Part 1
    source_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/GPi_project/FINAL/Part1/STN'
    dest_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/GPi_project/FINAL/Part1'
elif case == 2:
# Part 2
    source_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/GPi_project/FINAL/Part1/STN/NotPart1'
    dest_dir = '/home/udall-raid2/DBS_collaborators/DBS_for_orens/GPi_project/FINAL/Part2'

type_of_dirs = ['STN', 'SN', 'RN', '7T_SWI', '7T_T2']


# Work
# ----------------------------------------------------
Patients = sorted([f for f in os.listdir(os.path.join(dest_dir, 'Patients')) if not f.startswith('.')])

# For each patient in the destination directory
kk = 1
for patient in Patients:
    print('Patient {} - {}/{}'.format(patient, kk, len(Patients)))

    # Create segmentation folders and T2, SWI folders
    for type_dir in type_of_dirs:
        if not os.path.exists(os.path.join(dest_dir, 'Patients', patient, type_dir)):
            os.mkdir(os.path.join(dest_dir, 'Patients', patient, type_dir))
            print('{}: Created directory {}'.format(patient, type_dir))
            if type_dir == '7T_SWI' or type_dir == '7T_T2':
                if not os.path.exists(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Coronal')):
                    os.mkdir(os.path.join(dest_dir, 'Patients', patient, type_dir, 'Coronal'))
                    print('{}: Created sub-directory {}'.format(patient, type_dir + '/Coronal'))
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
                [shutil.copyfile(os.path.join(src_folder, f), os.path.join(dest_folder, 'Coronal', f)) for f in
                 os.listdir(src_folder) if f.find(type_dir) != -1]
            else:
                [shutil.copyfile(os.path.join(src_folder, f), os.path.join(dest_folder, f)) for f in os.listdir(src_folder)
                 if f.find(type_dir) != -1]

            print('{}: Copied {} files to {}'.format(patient, type_dir, dest_folder))
        except:
            print('{}: No files were copied'.format(patient))

    kk += 1



print('Done.')




