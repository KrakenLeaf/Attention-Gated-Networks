import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from shutil import copyfile
from models import get_model


def train(arguments):
    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(train_opts.arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup Data Loader - to RAM
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=train_opts.batchSize, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=0, batch_size=train_opts.batchSize, shuffle=False) # num_workers=16
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=0, batch_size=train_opts.batchSize, shuffle=False)

    # Setup the NN Model
    dataDims = [json_opts.training.batchSize, 1, train_dataset.image_dims[0], train_dataset.image_dims[1], train_dataset.image_dims[2]] # This is required only for the STN based network
    model = get_model(json_opts.model, dataDims)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Save the json configuration file to the checkpoints directory
    Ind = json_filename.rfind('/')
    copyfile(json_filename, os.path.join(json_opts.model.checkpoints_dir, json_opts.model.experiment_name, json_filename[Ind+1:]))

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels, _) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
        #for epoch_iter, (images, labels) in enumerate(train_loader, 1):
            # Make a training update
            model.set_input(images, labels) # Load data to GPU memory
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # So we won't increase lambda inside the epoch except for the first time
            model.haus_flag = False

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels, _) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # Visualise predictions
                visuals = model.get_current_visuals()
                #visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()

        # Update the Hausdorff distance lambda
        if (epoch + 1) % json_opts.model.haus_update_rate == 0:
            model.haus_flag = True
            print("Hausdorff distance lambda has been updated.")

# Workaround to run code from debug
class InputArgs:
    def __init__(self, json_file, debug):
        self.config = json_file
        self.debug = debug


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='CNN Seg Training Function')

	parser.add_argument('-c', '--config',  help='training config file', required=True)
	parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
	args = parser.parse_args()

	train(args)

