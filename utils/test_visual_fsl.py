'''
Test visually the manual segmentations w.r.t. the relevant scans
'''
import shutil
import os, sys
from subprocess import call
from shlex import split

# Parameters
# ---------------------------------------------
# We assume that in the below folder, there are "image" and "label" folders
db_path = '/home/naxos2-raid18/orens/DBS_for_orens/STN_project/T2_coronal_patients/db_stn_sn_rn_t2_71p/train'
image_type = 'T2_Coronal'

# Analysis
# ---------------------------------------------
images = sorted([f for f in os.listdir(os.path.join(db_path, 'image')) if f.find(image_type) != -1 and not f.startswith('.')])
labels = sorted([f for f in os.listdir(os.path.join(db_path, 'label')) if not f.startswith('.')])

assert (len(images) == len(labels)), "image and label length must be the same. Length of images: {}, length of labels: {}".format(len(images), len(labels))

for ii in range(len(images)):
    image = os.path.join(db_path, 'image', images[ii])
    label = os.path.join(db_path, 'label', labels[ii])

    print("Patient: {} label: {}".format(images[ii], labels[ii]))

    
    # Open in FSL
    call(split('fslview_deprecated %s %s' % (image, label)))




print('Done.')













