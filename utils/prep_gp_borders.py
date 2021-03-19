'''
Load GPi and GPe segementations and extract the estimated borders
'''

# Imports
# --------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import nibabel as nib
import re
from utils.measure_registration_results import remove_islands_gp

# Parameters
# --------------------------------------------------------------------------------------------------------------
in_folder = '/home/naxos2-raid18/orens/Projects/Seg_GPiGPe/AttentionUnet/checkpoints/experiment_unet_ct_dsv_mri_gpe94/mult_prj_probabilistic'

# functions
# --------------------------------------------------------------------------------------------------------------
def prep_borders(in_folder, max_cut=0.99, min_cut=0.01):
    # Load nifti files
    gpe_names = sorted([f for f in os.listdir(in_folder) if not f.startswith('.') and f.find('logit_class_1') != -1])
    gpi_names = sorted([f for f in os.listdir(in_folder) if not f.startswith('.') and f.find('logit_class_2') != -1])

    assert len(gpe_names) == len(gpi_names), "GPi and GPe number of files mismatch, exiting."

    # Extract borders for each patient
    for ii in range(len(gpe_names)):
        print("Working on {} and {}".format(gpe_names[ii], gpi_names[ii]))

        # Get id
        id = re.search('(.*)_logit', gpe_names[ii]).group(1)

        # Load data
        gpe = nib.load(os.path.join(in_folder, gpe_names[ii]))
        gpe_data = gpe.get_fdata()
        gpe_header = gpe.header
        #gpe_data = remove_islands_gp(gpe_data) # Clean

        gpi = nib.load(os.path.join(in_folder, gpi_names[ii]))
        gpi_data = gpi.get_fdata()
        gpi_header = gpi.header
        #gpi_data = remove_islands_gp(gpi_data) # Clean

        # Keep only the borders
        gpe_data[gpe_data <= min_cut] = 0
        gpe_data[gpe_data >= max_cut] = 0
        gpi_data[gpi_data <= min_cut] = 0
        gpi_data[gpi_data >= max_cut] = 0

        borders = gpe_data + gpi_data

        # Save nifti file
        nib.Nifti1Image(borders, affine=None, header=gpe_header).to_filename(
            os.path.join(in_folder, '{}_borders.nii.gz'.format(id)))

    return 0


# Main
# --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    prep_borders(in_folder)
    print("--- done ---")

