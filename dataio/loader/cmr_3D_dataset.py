import torch.utils.data as data
import numpy as np
import datetime
import re

from os import listdir
from os.path import join
from .utils import load_nifti_img, check_exceptions, is_image_file, load_nifti_lbl
import torchsample.transforms as ts

import math
from skimage.exposure import match_histograms
from skimage import exposure

class CMR3DDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(CMR3DDataset, self).__init__()
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')
        self.image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} Patients'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames]
            self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # # TODO: Make this a parameter
        # # This is a temporary workaround so the data will be of the same size
        # NumOfSlices = 176 #192 # Should be divisable by 2^4 = 16, as we have 4 maxpools in the encoder path
        # input  = input[:, :, 0:NumOfSlices]
        # target = target[:, :, 0:NumOfSlices]

        # handle exceptions
        check_exceptions(input, target)
        if self.transform:
            try:
                input, target = self.transform(input, target)
            except:
                # NOTE: Apparently there is a bug with adding additional channel, so I modified the code in that case
                dummy = input
                input  = self.transform(input, dummy)
                target = self.transform(target, dummy)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

'''
Modified class:
    1. Loads multi-label dataset - different labels are interpose (each label has different assigend value)
    2. Loads multi modal images  - all of the same projection. Images are assumed to be registered and interpolated 
                                   to the same grid
'''
class CMR3DDataset_MultiClass_MultiProj(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=True, modalities=['7T_T2'], rank=0):
        super(CMR3DDataset_MultiClass_MultiProj, self).__init__()

        # TODO: make this a parameter
        #self.TypeOfModal = ['7T_DTI_B0', '7T_DTI_FA'] # If we use B0 as well for the Thalamus seg

        ### ---
        #self.TypeOfModal = ['7T_T2']  # For T2 axial
        #self.TypeOfModal = ['7T_T2_cor'] # For T2 coronal
        #self.TypeOfModal = ['7T_SWI']
        self.TypeOfModal = modalities
        if rank == 0:
            print("Modalities: {}".format(self.TypeOfModal))

        # For now we assume all projections are axial - no coronal projections
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')

        self.image_filenames = []
        for mod in self.TypeOfModal:
            if mod == '7T_T2_cor':
                mod = '7T_T2'
            tmp_filenames = [join(image_dir, x) for x in listdir(image_dir) if (is_image_file(x) and x.find(mod) != -1)]
            self.image_filenames.append(sorted(tmp_filenames))
            # TODO: if mod == '7T_T2'
            if mod == '7T_T2' or mod == '7T_SWI': # This is the reference scan (all patients must have it) - use it to determine how many patients we have
                self.patient_len = len(tmp_filenames)
            elif mod == '7T_T1':
                # Secondary priority
                self.patient_len = len(tmp_filenames)
            elif mod == '7T_DTI_B0':
                # Tertiary priority
                self.patient_len = len(tmp_filenames)
            elif mod == '3T_T2':
                # Fourth priority
                self.patient_len = len(tmp_filenames)

        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        #assert len(self.image_filenames) == len(self.target_filenames)

        if rank == 0:
            print("\n".join(self.target_filenames))

        # Assume we always start from 7T_T2 Axial scans
        tmp_data = load_nifti_img(self.image_filenames[0][0], dtype=np.int16)
        self.image_dims = tmp_data[0].shape

        # report the number of images in the dataset
        if rank == 0:
            print('Number of {0} images: {1} Patients'.format(split, self.__len__()))

        # data augmentation
        # NOTE: in this case, disable the add dimension transform!
        #self.transform = transform
        self.transform = ts.Compose([
                                      ts.ToTensor(),
                                      ts.TypeCast(['float', 'long'])
                                    ])

        # data load into the ram memory
        self.t2_headers = []
        self.preload_data = preload_data
        if self.preload_data:
            if rank == 0:
                print('Preloading the {0} dataset ...'.format(split))
            #self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames] # Output is a list

            # Concatenate the raw data along the channels dimension
            self.raw_images = []
            for jj in range(len(self.image_filenames[0])):  # Per each patient, go over all modalities
                internal_cntr = 0
                for ii in range(len(self.image_filenames)):  # Go over all patients, left and right
                    #print('File: {}'.format(self.image_filenames[ii][jj])) # Only for DEBUG
                    if internal_cntr == 0: # First time
                        q_dat, tmp_header, _ = load_nifti_img(self.image_filenames[ii][jj], dtype=np.float32) # normalize values to [0,1] range

                        # if self.TypeOfModal[0] == '7T_T2_cor':
                        #     # Only for coronal slices
                        #     q_dat = np.transpose(q_dat, (0, 2, 1))

                        tmp_data = np.expand_dims(q_dat/np.max(q_dat.reshape(-1)), axis=0)
                        tmp_name = self.image_filenames[ii][jj]  # For the header file - identification in the multi GPU case
                    else: # Concatenate additional channels
                        q_dat = load_nifti_img(self.image_filenames[ii][jj], dtype=np.float32)[0] # normalize values to [0,1] range

                        # if self.TypeOfModal[0] == '7T_T2_cor':
                        #     # Only for coronal slices
                        #     q_dat = np.transpose(q_dat, (0, 2, 1))

                        concat_data = np.expand_dims(q_dat/np.max(q_dat.reshape(-1)), axis=0)
                        tmp_data = np.concatenate((tmp_data, concat_data), axis=0)
                    internal_cntr += 1

                # Add the concatenated multichannel data to the list
                self.raw_images.append(tmp_data)
                tmp_header['db_name'] = re.search('_P(.*).nii.gz', tmp_name).group(1)  # Data identifier
                self.t2_headers.append(tmp_header)

            # Load labels
            #self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            self.raw_labels = []
            for ii in self.target_filenames:
                label_tmp = load_nifti_img(ii, dtype=np.uint8)[0]

                # if self.TypeOfModal[0] == '7T_T2_cor':
                #     # Only for coronal slices
                #     label_tmp = np.transpose(label_tmp, (0, 2, 1))

                self.raw_labels.append(label_tmp)

            if rank == 0:
                print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        # NOTE: not implemented - use only preload_data = True
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])
            header = self.t2_headers[index]
            identifier = int(header['db_name'])

        # handle exceptions
        check_exceptions(input, target)
        # if self.transform:
        #     try:
        #         input, target = self.transform(input, target)
        #     except:
        #         # NOTE: Apparently there is a bug with adding additional channel, so I modified the code in that case
        #         #dummy = input
        #         input = self.transform(input)
        #         target = self.transform(target)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, identifier

    def __len__(self):
        #return len(self.image_filenames)
        return self.patient_len

'''
Modified class:
    T2 Axial and coronal registration
    1. Loads multi modal images  - all of the same projection. Images are assumed to be registered and interpolated 
                                   to the same grid
    2. Loads multi projections   - Loads in addition to multi modal images (all correspond to the same projection), 
                                   this module also loads data from different projections (i.e. if the axial and coronal 
                                   projections complement each other with different scan resolutions)
'''
class CMR3DDataset_t2_reg(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(CMR3DDataset_t2_reg, self).__init__()

        # TODO: make this a parameter
        self.TypeOfModal = ['7T_T2']
        self.TypeOfProj  = ['Axial', 'Coronal']

        # For now we assume all projections are axial - no coronal projections
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')

        self.image_filenames = []
        for mod in self.TypeOfModal:
            tmp_list = []
            for prj in self.TypeOfProj:
                tmp_str = [join(image_dir, x) for x in listdir(image_dir) if (is_image_file(x) and x.find(mod) != -1 and x.find(prj) != -1)]
                tmp_list.append(sorted(tmp_str))

                if mod == '7T_T2' and prj == 'Axial':
                    self.patient_len = len(tmp_str)

            self.image_filenames.append(tmp_list)

        # Assume we always start from 7T_T2 Axial scans
        tmp_data = load_nifti_img(self.image_filenames[0][0][0], dtype=np.int16)
        self.image_dims = tmp_data[0].shape

        self.target_filenames = [] # No labels for this project

        # report the number of images in the dataset
        print('Number of {0} images: {1} Patients'.format(split, self.__len__()))

        # data augmentation
        # NOTE: in this case, disable the add dimension transform!
        #self.transform = transform
        #self.transform = ts.TypeCast(['float', 'long'])
        self.transform = ts.Compose([
            ts.ToTensor(),
            ts.TypeCast(['float', 'long'])
        ])

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            #self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames] # Output is a list

            # Concatenate the raw data along the channels dimension
            self.raw_images = [] # This will be a list of lists
            # Per each patient, go over all modalities and projections
            #for jj in range(len(self.image_filenames[0][0])): # REAL
            for jj in range(len([0, 1])): # DEBUG
                tmp_data_list = []
                # Go over all projections
                for kk in range(len(self.image_filenames[0])):
                    internal_cntr = 0
                    # Go over all modalities (T2, T1, ...) of similar projection
                    for ii in range(len(self.image_filenames)):
                        # NOTE: Coronal data is already permuted in the correct directions
                        if self.image_filenames[ii][kk] != []: # Check if the data exists
                            #print(self.image_filenames[ii][kk][jj]) # For DEBUG
                            if internal_cntr == 0: # First time
                                q_dat = load_nifti_img(self.image_filenames[ii][kk][jj], dtype=np.float32)[0]  # normalize values to [0,1] range
                                q_dat = q_dat/np.max(q_dat.reshape(-1)) # Normalize
                                q_dat = self.zero_pad(q_dat)# zero pad
                                tmp_data = np.expand_dims(q_dat, axis=0)
                            else: # Concatenate additional channels
                                q_dat = load_nifti_img(self.image_filenames[ii][kk][jj], dtype=np.float32)[0]  # normalize values to [0,1] range
                                q_dat = q_dat/np.max(q_dat.reshape(-1))
                                q_dat = self.zero_pad(q_dat)
                                concat_data = np.expand_dims(q_dat, axis=0)
                                tmp_data = np.concatenate((tmp_data, concat_data), axis=0)
                            internal_cntr += 1

                    # Append for all modalities per same projection
                    tmp_data_list.append(tmp_data)

                # Add the concatenated multichannel data to the list
                # [0] - Axial, [1] - Coronal
                self.raw_images.append(tmp_data_list)

            self.raw_labels = [f for f in range(len(self.image_filenames)+1)] # Dummy: no labels for this project
            print('Loading is done\n')

    # Do zero padding so that the volume (axial or coronal) will be a cube
    def zero_pad(self, vol):
        dims = vol.shape # XYZ
        max_size = np.max(dims)
        if dims[2] < dims[1]: # Axial volume
            num_to_pad = max_size - dims[2]
            if num_to_pad % 2 == 0:
                # num_to_pad is even
                pad_before = int(num_to_pad / 2)
                pad_after = int(num_to_pad / 2)
            else:
                # num_to_pad is odd
                pad_before = int(num_to_pad / 2)
                pad_after = int(num_to_pad / 2) + 1

            vol_pad = np.pad(vol, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
        elif dims[1] < dims[2]: # Coronal volume
            num_to_pad = max_size - dims[1]
            if num_to_pad % 2 == 0:
                # num_to_pad is even
                pad_before = int(num_to_pad / 2)
                pad_after = int(num_to_pad / 2)
            else:
                # num_to_pad is odd
                pad_before = int(num_to_pad / 2)
                pad_after = int(num_to_pad / 2) + 1

            vol_pad = np.pad(vol, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')

        # Perform this check - just to make sure there are no errors in the code
        vol_pad_dims = vol_pad.shape
        assert vol_pad_dims[0] == vol_pad_dims[1], "zero_pad: volume is not cube: {}, {}, {}".format(vol_pad_dims[0],
                                                                                                     vol_pad_dims[1],
                                                                                                     vol_pad_dims[2])
        assert vol_pad_dims[0] == vol_pad_dims[2], "zero_pad: volume is not cube: {}, {}, {}".format(vol_pad_dims[0],
                                                                                                     vol_pad_dims[1],
                                                                                                     vol_pad_dims[2])
        assert vol_pad_dims[1] == vol_pad_dims[2], "zero_pad: volume is not cube: {}, {}, {}".format(vol_pad_dims[0],
                                                                                                     vol_pad_dims[1],
                                                                                                     vol_pad_dims[2])
        return vol_pad

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        # NOTE: not implemented - use only preload_data = True
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = []
            for ii in range(len(self.raw_images[0])):
                input.append(np.copy(self.raw_images[index][ii]))
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        #check_exceptions(input, target)
        # if self.transform:
        #     try:
        #         input, target = self.transform(input, target)
        #     except:
        #         # NOTE: Apparently there is a bug with adding additional channel, so I modified the code in that case
        #         dummy = input
        #         input  = self.transform(input, dummy)
        #         target = self.transform(target, dummy)

        if self.transform:
            for ii in range(len(input)):
                input[ii] = self.transform(input[ii])
            target = self.transform(target)

        return input, target

    def __len__(self):
        # return len(self.image_filenames)
        return self.patient_len

'''
Modified class:
    1. Loads multi-label dataset - different labels are interpose (each label has different assigend value)
    2. Loads multi modal images  - all of the same projection. Images are assumed to be registered and interpolated 
                                   to the same grid
'''
class CMR3DDataset_MultiClass_MultiProj_unreg(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=True):
        super(CMR3DDataset_MultiClass_MultiProj_unreg, self).__init__()

        # TODO: make this a parameter
        self.TypeOfModal = ['7T_T2', '7T_T1', '7T_DTI_FA']
        #self.TypeOfModal = ['7T_T2']

        # For now we assume all projections are axial - no coronal projections
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')

        self.image_filenames = []
        for mod in self.TypeOfModal:
            tmp_filenames = [join(image_dir, x) for x in listdir(image_dir) if (is_image_file(x) and x.find(mod) != -1)]
            self.image_filenames.append(sorted(tmp_filenames))
            # TODO: if mod == '7T_T2'
            if mod == '7T_T2': # This is the reference scan (all patients must have it) - use it to determine how many patients we have
                self.patient_len = len(tmp_filenames)

        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        #assert len(self.image_filenames) == len(self.target_filenames)

        # Assume we always start from 7T_T2 Axial scans
        tmp_data, meta = load_nifti_img(self.image_filenames[0][0], dtype=np.int16)
        #self.image_dims = tmp_data[0].shape
        self.image_dims = tmp_data.shape

        # report the number of images in the dataset
        print('Number of {0} images: {1} Patients'.format(split, self.__len__()))

        # data augmentation
        # NOTE: in this case, disable the add dimension transform!
        #self.transform = transform
        self.transform = ts.Compose([
                                      ts.ToTensor(),
                                      ts.TypeCast(['float', 'long'])
                                    ])

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            #self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames] # Output is a list

            # Concatenate the raw data along the channels dimension
            self.raw_images = []
            for jj in range(len(self.image_filenames[0])):  # Per each patient, go over all modalities
                tmp_data = []
                for ii in range(len(self.image_filenames)):  # Go over all patients
                    #print('File: {}'.format(self.image_filenames[ii][jj])) # Only for DEBUG
                    q_dat = load_nifti_img(self.image_filenames[ii][jj], dtype=np.float32)[0] # normalize values to [0,1] range
                    tmp_data.append(np.expand_dims(q_dat/np.max(q_dat.reshape(-1)), axis=0))

                # Add the concatenated multichannel data to the list
                self.raw_images.append(tmp_data)

            self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        # NOTE: not implemented - use only preload_data = True
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = []
            for ii in range(len(self.raw_images[0])):
                input.append(np.copy(self.raw_images[index][ii]))
            target = np.copy(self.raw_labels[index])


        # handle exceptions
        #check_exceptions(input, target)
        # if self.transform:
        #     try:
        #         input, target = self.transform(input, target)
        #     except:
        #         # NOTE: Apparently there is a bug with adding additional channel, so I modified the code in that case
        #         #dummy = input
        #         input = self.transform(input)
        #         target = self.transform(target)

        if self.transform:
            for ii in range(len(input)):
                input[ii] = self.transform(input[ii])
            target = self.transform(target)

        return input, target

    def __len__(self):
        #return len(self.image_filenames)
        return self.patient_len

'''
Modified class:
    1. Loads multi-label dataset - different labels are interpose (each label has different assigend value)
    2. Loads multi modal images  - all of the same projection. Images are assumed to be registered and interpolated 
                                   to the same grid
    3. Loads multi projections   - Loads in addition to multi modal images (all correspond to the same projection), 
                                   this module also loads data from different projections (i.e. if the axial and coronal 
                                   projections complement each other with different scan resolutions)
'''
class CMR3DDataset_MultiClass_MultiProj_V2(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=True, rank=0, world_size=1):
        super(CMR3DDataset_MultiClass_MultiProj_V2, self).__init__()

        # TODO: make this an external parameter?
        internal_hist_augmentation_flag = 0

        # TODO: make this an external parameter?
        #self.TypeOfModal = ['7T_T2', '7T_T1', '7T_DTI_FA']
        self.TypeOfModal = ['7T_T2']
        self.TypeOfProj  = ['Axial', 'Coronal']

        # For now we assume all projections are axial - no coronal projections
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')

        self.image_filenames = []
        for mod in self.TypeOfModal:
            tmp_list = []
            for prj in self.TypeOfProj:
                tmp_str = [join(image_dir, x) for x in listdir(image_dir) if (is_image_file(x) and x.find(mod) != -1 and x.find(prj) != -1)]
                tmp_list.append(sorted(tmp_str))

                # if mod == '7T_T2' and prj == 'Axial':
                #     self.patient_len = len(tmp_str)

            self.image_filenames.append(tmp_list)

        # Assume we always start from 7T_T2 Axial scans
        tmp_data = load_nifti_img(self.image_filenames[0][0][0], dtype=np.int16)
        self.image_dims = tmp_data[0].shape

        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        #assert len(self.image_filenames) == len(self.target_filenames)

        # Divide data to each rank
        grp_size = math.ceil(len(self.target_filenames) / world_size)
        self.image_filenames[0][0] = self.image_filenames[0][0][rank * grp_size:(rank + 1) * grp_size]
        self.image_filenames[0][1] = self.image_filenames[0][1][rank * grp_size:(rank + 1) * grp_size]
        self.target_filenames = self.target_filenames[rank * grp_size:(rank + 1) * grp_size]

        self.patient_len = len(self.target_filenames)

        # print("len(self.target_filenames (rank {}) = {})".format(rank, len(self.target_filenames )))
        # print(self.image_filenames[0][1])

        # report the number of images in the dataset
        print('Number of {0} images: {1} Patients'.format(split, self.__len__()))

        # data augmentation
        # NOTE: in this case, disable the add dimension transform!
        #self.transform = transform
        #self.transform = ts.TypeCast(['float', 'long'])
        self.transform = ts.Compose([
            ts.ToTensor(),
            ts.TypeCast(['float', 'long'])
        ])

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            #self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames] # Output is a list

            # Concatenate the raw data along the channels dimension
            self.raw_images = [] # This will be a list of lists
            # Per each patient, go over all modalities and projections
            for jj in range(len(self.image_filenames[0][0])):
            #for jj in range(len([1])):
                tmp_data_list = []
                # Go over all projections
                for kk in range(len(self.image_filenames[0])):
                    internal_cntr = 0
                    # Go over all modalities (T2, T1, ...) of similar projection
                    for ii in range(len(self.image_filenames)):
                        # NOTE: Coronal data is already permuted in the correct directions
                        if self.image_filenames[ii][kk] != []: # Check if the data exists
                            #print(self.image_filenames[ii][kk][jj]) # For DEBUG
                            if internal_cntr == 0: # First time
                                q_dat = load_nifti_img(self.image_filenames[ii][kk][jj], dtype=np.float32)[0]  # normalize values to [0,1] range
                                q_dat = q_dat / np.max(q_dat.ravel()) # Normalize

                                # Do image histogram augmentation
                                if internal_hist_augmentation_flag == 1:
                                    q_dat = adaptive_hist_aug(q_dat)

                                ### TEST
                                #q_dat = q_dat[96-32:96+32, 96-32:96+32, 96-32:96+32]
                                ##########

                                tmp_data = np.expand_dims(q_dat, axis=0)
                            else: # Concatenate additional channels
                                q_dat = load_nifti_img(self.image_filenames[ii][kk][jj], dtype=np.float32)[0]  # normalize values to [0,1] range
                                q_dat / np.max(q_dat.ravel())

                                # Do image histogram augmentation
                                if internal_hist_augmentation_flag == 1:
                                    q_dat = adaptive_hist_aug(q_dat)

                                ### TEST
                                #q_dat = q_dat[96-32:96+32, 96-32:96+32, 96-32:96+32]
                                ##########

                                concat_data = np.expand_dims(q_dat, axis=0)
                                tmp_data = np.concatenate((tmp_data, concat_data), axis=0)
                            internal_cntr += 1

                    # Append for all modalities per same projection
                    tmp_data_list.append(tmp_data)

                # Add the concatenated multichannel data to the list
                # [0] - Axial, [1] - Coronal
                self.raw_images.append(tmp_data_list)

            self.raw_labels = [load_nifti_lbl(ii, dtype=np.uint8)[0] for ii in self.target_filenames] # Round the labels as well
            # ### TEST
            # self.raw_labels = []
            # for ii in self.target_filenames:
            #     tmp_tmp = load_nifti_img(ii, dtype=np.uint8)[0]
            #     self.raw_labels.append(tmp_tmp[96-32:96+32, 96-32:96+32, 96-32:96+32])
            # ##########

            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        # NOTE: not implemented - use only preload_data = True
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = []
            for ii in range(len(self.raw_images[0])):
                input.append(np.copy(self.raw_images[index][ii]))
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        #check_exceptions(input, target)
        # if self.transform:
        #     try:
        #         input, target = self.transform(input, target)
        #     except:
        #         # NOTE: Apparently there is a bug with adding additional channel, so I modified the code in that case
        #         dummy = input
        #         input  = self.transform(input, dummy)
        #         target = self.transform(target, dummy)

        if self.transform:
            for ii in range(len(input)):
                input[ii] = self.transform(input[ii])
            target = self.transform(target)

        return input, target

    def __len__(self):
        return self.patient_len

class CMR3DDataset_MultiClass(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(CMR3DDataset_MultiClass, self).__init__()
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')
        self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} Patients'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames]
            self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            print('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # TODO: Make this a parameter
        # This is a temporary workaround so the data will be of the same size
        NumOfSlices = 176  # 192 # Should be divisable by 2^4 = 16, as we have 4 maxpools in the encoder path
        input = input[:, :, 0:NumOfSlices]
        target = target[:, :, 0:NumOfSlices]

        # handle exceptions
        check_exceptions(input, target)
        if self.transform:
            try:
                input, target = self.transform(input, target)
            except:
                # NOTE: Apparently there is a bug with adding additional channel, so I modified the code in that case
                dummy = input
                input = self.transform(input, dummy)
                target = self.transform(target, dummy)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

def adaptive_hist_aug(data_in):
    data_out = np.zeros(data_in.shape)
    for ii in range(data_out.shape[2]):
        data_out[:, :, ii] = exposure.equalize_adapthist(data_in[:, :, ii], clip_limit=0.03)

        # p2, p98 = np.percentile(data_in[:, :, ii], (2, 98))
        # data_out[:, :, ii] = exposure.rescale_intensity(data_in[:, :, ii], in_range=(p2, p98))

        # data_out[:, :, ii] = exposure.equalize_hist(data_in[:, :, ii])


        # Save
    internal_debug_flag = 1
    if internal_debug_flag == 1:
        import SimpleITK as sitk
        import os
        label_img = sitk.GetImageFromArray(np.transpose(data_out, (2, 1, 0)))
        label_img.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, 1])
        sitk.WriteImage(label_img, os.path.join('.', 'test_aug.nii.gz'))

    return data_out