'''
Check if T2 / SWI foders have mistakenly segmentation files in them
'''
import os

db_folder = '/home/naxos2-raid18/orens/DBS_for_orens/STN_project/T2_coronal_patients/Patients'

patients = sorted([f for f in os.listdir(db_folder) if not f.startswith('.')])

for patient in patients:
    swi_dir = os.path.join(db_folder, patient, '7T_SWI', 'Coronal')
    t2_dir = os.path.join(db_folder, patient, '7T_T2', 'Coronal')

    swi_files = sorted([f for f in os.listdir(swi_dir) if not f.startswith('.') and
                        (f.find('STN') != -1 or
                         f.find('SN') != -1 or
                         f.find('RN') != -1
                         )])
    t2_files = sorted([f for f in os.listdir(t2_dir) if not f.startswith('.') and
                        (f.find('STN') != -1 or
                         f.find('SN') != -1 or
                         f.find('RN') != -1
                         )])

    if swi_files:
        print("Patient {} has segmentations in SWI folder".format(patient))

    if t2_files:
        print("Patient {} has segmentations in T2 folder".format(patient))


print('Done.')









