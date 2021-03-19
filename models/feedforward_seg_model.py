import torch
from torch.autograd import Variable
import torch.optim as optim

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor

import torch.nn.functional as F

import torch.nn as nn

class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample, rank=opts.rank)
        if self.use_cuda: self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            # print the network details
            if kwargs.get('verbose', True):
                if opts.rank == 0:
                    print('Network is initialized')
                    print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                #assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    def set_input_multi(self, rank, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.to(rank) if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.to(rank)) if self.use_cuda else Variable(_input)
                #assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    def forward(self, split):
        if split == 'train':
            self.prediction = self.net(Variable(self.input)) # 16 bit float
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input)) # 16 bit float
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)
            
    def backward(self):
        #self.loss_S = self.criterion(self.prediction, self.target) # OS: original, no evolving lambdas
        self.loss_S = self.criterion(self.prediction, self.target, self.haus_flag) # OS: Evolving lambdas
        self.loss_S.backward()
        #self.loss_S.backward(retain_graph=True)## OS: test

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        #self.loss_S = self.criterion(self.prediction, self.target) # OS: original, no evolving lambdas
        self.loss_S = self.criterion(self.prediction, self.target, self.haus_flag)  # OS: Evolving lambdas

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        #return OrderedDict([('Seg_Loss', self.loss_S.data[0])
        #                    ])
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input, 'img')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time (self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)


# OS: For multi projection - unregistered
class FeedForwardSegmentation_unreg(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation_MultProj'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        if hasattr(opts, 'dataDims') == False:
            self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                                   in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                                   tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                                   attention_dsample=opts.attention_dsample)
        else: # For STN based network
            self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                                   in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                                   tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                                   attention_dsample=opts.attention_dsample,
                                   dataDims=opts.dataDims)
        if self.use_cuda: self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    # OS: for miult projection axial and coronal
    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input[0].size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                for ii in range(len(_input)):
                    _input[ii] = _input[ii].permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = []
                for ii in range(len(_input)):
                    self.input.append(_input[ii].cuda()) if self.use_cuda else _input[ii]
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: This is the original function, I have modified the above function to handle the multi projection case
    # Change name for original behavior
    def set_input_original(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: this is for axial and coronal data
    def forward(self, split):
        if split == 'train':
            # Axial and Coronal
            self.prediction = self.net(Variable(self.input[0]), Variable(self.input[1]), Variable(self.input[2]))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input[0]), Variable(self.input[1]), Variable(self.input[2]))
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    # OS: Origianl - only axial data
    def forward_original(self, split):
        if split == 'train':
            # Axial and Coronal
            self.prediction = self.net(Variable(self.input))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input))
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    def backward(self):
        # self.loss_S = self.criterion(self.prediction, self.target) # OS: non-evolving laambdas
        self.loss_S = self.criterion(self.prediction, self.target, self.haus_flag) # Evolving lambdas
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        #self.loss_S = self.criterion(self.prediction, self.target) # Non-evolving lambdas
        self.loss_S = self.criterion(self.prediction, self.target, self.haus_flag)  # Evolving lambdas

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        # return OrderedDict([('Seg_Loss', self.loss_S.data[0])
        #                    ])
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input[0], 'img') # OS: added [0] to handle the multiprojection case : T2 axial
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time(self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp / float(bsize), bp / float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)


# OS: For multi projection
class FeedForwardSegmentation_multproj(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation_MultProj'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        if hasattr(opts, 'dataDims') == False:
            self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                                   in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                                   tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                                   attention_dsample=opts.attention_dsample)
        else: # For STN based network
            self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                                   in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                                   tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                                   attention_dsample=opts.attention_dsample,
                                   dataDims=opts.dataDims, rank=opts.rank)

        if self.use_cuda: self.net = self.net.cuda() # OS - original, might be redundant

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    # OS: for multi projection axial and coronal
    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input[0].size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                for ii in range(len(_input)):
                    _input[ii] = _input[ii].permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = []
                for ii in range(len(_input)):
                    self.input.append(_input[ii].cuda()) if self.use_cuda else _input[ii]
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: for multi projection axial and coronal
    def set_input_multi(self, rank, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input[0].size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                for ii in range(len(_input)):
                    _input[ii] = _input[ii].permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2],
                                                                                     bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = []
                for ii in range(len(_input)):
                    self.input.append(_input[ii].to(rank)) if self.use_cuda else _input[ii]
            elif idx == 1:
                self.target = Variable(_input.to(rank)) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: This is the original function, I have modified the above function to handle the multi projection case
    # Change name for original behavior
    def set_input_original(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: this is for axial and coronal data
    def forward(self, split):
        if split == 'train':
            # Axial and Coronal
            self.prediction = self.net(Variable(self.input[0]), Variable(self.input[1]))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input[0]), Variable(self.input[1]))
                # Apply a softmax and return a segmentation map
                #self.logits = self.net.apply_argmax_softmax(self.prediction) # OS - original
                self.logits = F.softmax(self.prediction, dim=1)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    # OS: Origianl - only axial data
    def forward_original(self, split):
        if split == 'train':
            # Axial and Coronal
            self.prediction = self.net(Variable(self.input))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input))
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    def backward(self, rank=0):
        self.loss_S = self.criterion(rank, self.prediction, self.target, self.haus_flag)
        self.loss_S.backward()

    def optimize_parameters(self, rank=0):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward(rank)
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self, rank=0):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(rank, self.prediction, self.target, self.haus_flag)

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_segmentation_stats_multi(self):
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        return self.seg_scores, self.dice_score

    def get_current_errors(self):
        # return OrderedDict([('Seg_Loss', self.loss_S.data[0])
        #                    ])
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def get_current_errors_multi(self):
        #return torch.tensor(self.loss_S.data.item()).to(rank)
        return self.loss_S

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input[0], 'img') # OS: added [0] to handle the multiprojection case : T2 axial
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time(self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp / float(bsize), bp / float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)


# OS: For T2 axial - coronal registration
class FeedForward_registration(BaseModel):

    def name(self):
        return 'FeedForward_registration'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        if hasattr(opts, 'dataDims') == False:
            self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                                   in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                                   tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                                   attention_dsample=opts.attention_dsample)
        else: # For STN based network
            self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                                   in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                                   tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                                   attention_dsample=opts.attention_dsample,
                                   dataDims=opts.dataDims)
        if self.use_cuda: self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    # OS: for miult projection axial and coronal
    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input[0].size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                for ii in range(len(_input)):
                    _input[ii] = _input[ii].permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = []
                for ii in range(len(_input)):
                    self.input.append(_input[ii].cuda()) if self.use_cuda else _input[ii]
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: This is the original function, I have modified the above function to handle the multi projection case
    # Change name for original behavior
    def set_input_original(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                # assert self.input.size() == self.target.size() # Irrelevant for the multimodal case

    # OS: this is for axial and coronal data
    def forward(self, split):
        if split == 'train':
            # Axial and Coronal
            self.prediction = self.net(Variable(self.input[0]), Variable(self.input[1]))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input[0]), Variable(self.input[1]))
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    # OS: Origianl - only axial data
    def forward_original(self, split):
        if split == 'train':
            # Axial and Coronal
            self.prediction = self.net(Variable(self.input))
        elif split == 'test':
            with torch.no_grad():
                self.prediction = self.net(Variable(self.input))
                # Apply a softmax and return a segmentation map
                self.logits = self.net.apply_argmax_softmax(self.prediction)
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    def backward(self):
        # self.loss_S = self.criterion(self.prediction, self.target) # OS: non-evolving lambdas
        self.loss_S = self.criterion(self.prediction, self.target, self.haus_flag) # Evolving lambdas
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0: self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        #self.loss_S = self.criterion(self.prediction, self.target) # Non-evolving lambdas
        self.loss_S = self.criterion(self.prediction, self.target, self.haus_flag)  # Evolving lambdas

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        # return OrderedDict([('Seg_Loss', self.loss_S.data[0])
        #                    ])
        return OrderedDict([('Seg_Loss', self.loss_S.data.item())
                            ])

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input[0], 'img') # OS: added [0] to handle the multiprojection case : T2 axial
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time(self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp / float(bsize), bp / float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)
