'''
This script adds 7T_T2 scans to the 3Tseg_project folder
'''

import os, sys
import numpy as np
import shutil
from subprocess import call
from shlex import split

# -- Parameters --
src_folder = '/home/naxos2-raid18/orens/DBS_for_orens/DiseaseClassification/Patients'
patients_folder = '/home/naxos2-raid18/orens/DBS_for_orens/3Tseg_project/Testing/Patients'

patients = sorted([f for f in os.listdir(patients_folder) if not f.startswith('.') and
                   f.find('DYS') == -1 and f.find('ET012') == -1 and f.find('P033') == -1])

#patients = ['P033']

for patient in patients:
    print('Working on patient {}'.format(patient))
    print('-------------------------------------')

    if not os.path.exists(os.path.join(patients_folder, patient, '7T_T2')):
        os.mkdir(os.path.join(patients_folder, patient, '7T_T2'))
    folder_7t = os.path.join(patients_folder, patient, '7T_T2', 'Axial')
    if not os.path.exists(folder_7t):
        os.mkdir(folder_7t)
        print("Created folder {}".format(folder_7t))

    src_pat = os.path.join(src_folder, patient, '7T_T2', 'Axial')
    file_to_copy = [f for f in os.listdir(src_pat) if f.find('nii') != -1 and f.find('_iso') == -1 and f.find('bet') == -1][0]

    # Copy the file
    shutil.copyfile(os.path.join(src_pat, file_to_copy), os.path.join(folder_7t, file_to_copy))
    print('Copied file {}'.format(os.path.join(src_pat, file_to_copy)))

    # Resample to 0.5 mm isotropic grid (Bspline interpolation)
    input_name = os.path.join(src_pat, file_to_copy)
    output_name = os.path.join(folder_7t, '_iso_resampled_bspline_' + file_to_copy)
    call(split('ResampleImageBySpacing 3 %s %s 0.5 0.5 0.5 0 0 4' % (input_name, output_name)))

print('- done -')









