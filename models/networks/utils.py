import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks_other import init_weights
from models.deformable_convs.deform_conv_test2 import DeformConv3D

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1), init_dilation=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size, dilation=init_dilation),
                                       nn.BatchNorm3d(out_size, track_running_stats=True), # OS
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size, dilation=init_dilation),
                                       nn.BatchNorm3d(out_size, track_running_stats=True),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size, idilation=nit_dilation),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size, dilation=init_dilation),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

# OS: Deformable 3D convolutions
class UnetConv3_deformable(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=3, padding_size=1):
        super(UnetConv3_deformable, self).__init__()
        # NOTE: For simplicity, this module assumes isotropic kernels and padding

        # Offsets are learnt by a 3D convolution layer (can be a more complex network)
        self.offsets = nn.Conv3d(in_size, 3 * kernel_size ** 3, kernel_size=kernel_size, padding=padding_size)

        # off_size = 3 * kernel_size ** 3
        # self.offsets = nn.Sequential(nn.Conv3d(in_size, off_size, kernel_size=kernel_size, padding=padding_size),
        #                              nn.BatchNorm3d(off_size, track_running_stats=True), # OS
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv3d(off_size, off_size, kernel_size=kernel_size,padding=padding_size),
        #                              nn.BatchNorm3d(off_size, track_running_stats=True),  # OS
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv3d(off_size, off_size, kernel_size=kernel_size,padding=padding_size),
        #                              nn.BatchNorm3d(off_size, track_running_stats=True),  # OS
        #                              nn.ReLU(inplace=True),)

        self.deform_conv = DeformConv3D(in_size, out_size, kernel_size=kernel_size, padding=padding_size)

        if is_batchnorm:
            self.relu = nn.Sequential(nn.BatchNorm3d(out_size, track_running_stats=True), # OS
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size, track_running_stats=True),
                                       nn.ReLU(inplace=True),)
        else:
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        #internal_flag = 0
        for m in self.children():
            # if internal_flag == 0:
            #     # Zero weights initialization for the offsets weights - according to Jifeng et. al "Deformable Convolutional Networks"
            #     internal_flag += 1
            #     m.weight.data = torch.zeros_like(m.weight.data)
            # else:
            if not isinstance(m, DeformConv3D): # DeformConv3D is initialized from inside its init
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Calculate offsets
        offsets = self.offsets(inputs)

        # Convolutions
        outputs = self.relu(self.deform_conv(inputs, offsets))
        outputs = self.conv2(outputs)
        return outputs, offsets

# OS: Dense block implementation for 3D convolutions
class DenseBlock3D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1), depth=3):
        super(DenseBlock3D, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm3d(num_features=out_size, track_running_stats=True)

        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.init_stride = init_stride

        self.is_batchnorm = is_batchnorm

        # Depth of the dense block
        # -----------------------------------
        if depth < 1:
            depth = 1 # Minimal depth is 1
        self.dense_depth = depth

        # Initial convolution operation on the input data
        self.init_conv = nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size)

        # Define further convolutions in a for loop
        self.ops_dict = {}
        for ii in range(depth - 1):
            name = "op_{}".format(ii + 1)
            self.ops_dict[name] = nn.Conv3d((ii + 1) * out_size, out_size, kernel_size, init_stride, padding_size).cuda()

        self.batchnorm = nn.BatchNorm3d(out_size, track_running_stats=True)

        if is_batchnorm:
            # self.conv1 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
            #                            nn.BatchNorm3d(out_size),
            #                            nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size, track_running_stats=True),
                                       nn.ReLU(inplace=True),)
        else:
            # self.conv1 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
            #                            nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        #outputs = self.conv1(inputs)
        #outputs = self.conv2(outputs)

        # Create 3D convolution blocks and concatenate along the channels dimension
        dense = self.init_conv(inputs)
        for ii in range(self.dense_depth - 1):
            name = "op_{}".format(ii + 1)

            # Batch (0) x Channels (1) x W x H x D
            outcnn = self.relu(self.ops_dict[name](dense))

            # Concatenate and relu
            if ii == self.dense_depth - 2:
                dense = self.relu(outcnn)
            else:
                dense = self.relu(torch.cat([dense, outcnn], 1))

        # Apply batch normalization if required
        if self.is_batchnorm:
            dense = self.relu(self.batchnorm(dense))

        # Apply last convolution
        outputs = self.conv2(dense)

        return outputs

# OS: Spatial Transformer Block
class STNblock(nn.Module):
    def __init__(self, n_channels=1, input_grid_size=[96,48,192], target_grid_size=[], type='3d'):
        super(STNblock, self).__init__()
        '''
        n_channels - number of input channels
        type - '2d' 2D (2 x 3) affine transformations, 
               '3d' 3D (3 x 4) affine transformations.
        '''

        # Output grid size
        self.target_grid_size = target_grid_size
        out_size = 8

        # Spatial transformer localization-network
        # TODO: this can be replaced with i.e. a dense block
        if type.lower() == '2d':
            self.localization = nn.Sequential(
                nn.Conv2d(n_channels, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        elif type.lower() == '3d':
            self.localization = nn.Sequential(
                nn.Conv3d(n_channels, 8, kernel_size=7),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True),
                nn.Conv3d(8, 10, kernel_size=5),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True)
            )

        FCdims = 70400 #int(out_size * input_grid_size[0] * input_grid_size[1] * input_grid_size[2] / 2**4)
        FCfinal = 32

        # Regressor for the 3 * 2 affine matrix - outputs the /theta vectors
        if type.lower() == '2d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 3 * 2)
            )
            self.dims = [2, 3]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif type.lower() == '3d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 4 * 3)
            )
            self.dims = [3, 4]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # print("xs size = {}".format(xs.size()))
        if x.dim() == 5:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4]) # This is for 3D only
        else:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])
        theta = self.fc_loc(xs)
        # print("theta size = {}".format(theta.size()))
        theta = theta.view(-1, self.dims[0], self.dims[1])
        # print("theta size = {}".format(theta.size()))

        grid = F.affine_grid(theta, self.target_grid_size, align_corners=False)
        # print("grid size = {}".format(grid.size()))
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        return self.stn(x)

class STNblock_v2(nn.Module):
    def __init__(self, n_channels=1, input_grid_size=[96,48,192], target_grid_size=[], type='3d'):
        super(STNblock_v2, self).__init__()
        '''
        n_channels - number of input channels
        type - '2d' 2D (2 x 3) affine transformations, 
               '3d' 3D (3 x 4) affine transformations.
        '''

        # Output grid size
        self.target_grid_size = target_grid_size
        out_size = 8

        # Spatial transformer localization-network
        # TODO: this can be replaced with i.e. a dense block
        if type.lower() == '2d':
            self.localization = nn.Sequential(
                nn.Conv2d(n_channels, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        elif type.lower() == '3d':
            self.localization = nn.Sequential(
                nn.Conv3d(n_channels, 8, kernel_size=7),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True),
                nn.Conv3d(8, 10, kernel_size=5),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True)
            )

        FCdims = 211200
        FCfinal = 32

        # Regressor for the 3 * 2 affine matrix - outputs the /theta vectors
        if type.lower() == '2d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 3 * 2)
            )
            self.dims = [2, 3]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif type.lower() == '3d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 4 * 3)
            )
            self.dims = [3, 4]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # print("xs size = {}".format(xs.size()))
        if x.dim() == 5:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4]) # This is for 3D only
        else:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])
        theta = self.fc_loc(xs)
        # print("theta size = {}".format(theta.size()))
        theta = theta.view(-1, self.dims[0], self.dims[1])
        # print("theta size = {}".format(theta.size()))

        grid = F.affine_grid(theta, self.target_grid_size, align_corners=False)
        # print("grid size = {}".format(grid.size()))
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        return self.stn(x)

################################################################################################
########################### STN bolcks for unregistered T1 and FA ##############################
################################################################################################
class STNblock_t1_v2(nn.Module):
    def __init__(self, n_channels=1, input_grid_size=[96,48,192], target_grid_size=[], type='3d'):
        super(STNblock_t1_v2, self).__init__()
        '''
        n_channels - number of input channels
        type - '2d' 2D (2 x 3) affine transformations, 
               '3d' 3D (3 x 4) affine transformations.
        '''

        # Output grid size
        self.target_grid_size = target_grid_size
        out_size = 8

        # Spatial transformer localization-network
        # TODO: this can be replaced with i.e. a dense block
        if type.lower() == '2d':
            self.localization = nn.Sequential(
                nn.Conv2d(n_channels, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        elif type.lower() == '3d':
            self.localization = nn.Sequential(
                nn.Conv3d(n_channels, 8, kernel_size=7),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True),
                nn.Conv3d(8, 10, kernel_size=5),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True)
            )

        FCdims = 81920
        FCfinal = 32

        # Regressor for the 3 * 2 affine matrix - outputs the /theta vectors
        if type.lower() == '2d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 3 * 2)
            )
            self.dims = [2, 3]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif type.lower() == '3d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 4 * 3)
            )
            self.dims = [3, 4]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # print("xs size = {}".format(xs.size()))
        if x.dim() == 5:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4]) # This is for 3D only
        else:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])
        theta = self.fc_loc(xs)
        # print("theta size = {}".format(theta.size()))
        theta = theta.view(-1, self.dims[0], self.dims[1])
        #print("T1: theta = {}".format(theta))

        grid = F.affine_grid(theta, self.target_grid_size, align_corners=False)
        # print("grid size = {}".format(grid.size()))
        x = F.grid_sample(x, grid, align_corners=False)

        return x, theta

    def forward(self, x):
        out, affine = self.stn(x)
        return out, affine

class STNblock_fa_v2(nn.Module):
    def __init__(self, n_channels=1, input_grid_size=[96,48,192], target_grid_size=[], type='3d'):
        super(STNblock_fa_v2, self).__init__()
        '''
        n_channels - number of input channels
        type - '2d' 2D (2 x 3) affine transformations, 
               '3d' 3D (3 x 4) affine transformations.
        '''

        # Output grid size
        self.target_grid_size = target_grid_size
        out_size = 8

        # Spatial transformer localization-network
        # TODO: this can be replaced with i.e. a dense block
        if type.lower() == '2d':
            self.localization = nn.Sequential(
                nn.Conv2d(n_channels, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        elif type.lower() == '3d':
            self.localization = nn.Sequential(
                nn.Conv3d(n_channels, 8, kernel_size=7),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True),
                nn.Conv3d(8, 10, kernel_size=5),
                nn.MaxPool3d(2, stride=2),
                nn.ReLU(True)
            )

        FCdims = 8640
        FCfinal = 32

        # Regressor for the 3 * 2 affine matrix - outputs the /theta vectors
        if type.lower() == '2d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 3 * 2)
            )
            self.dims = [2, 3]

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif type.lower() == '3d':
            self.fc_loc = nn.Sequential(
                nn.Linear(FCdims, FCfinal), # TODO: get this number 211200 automatically
                nn.ReLU(True),
                nn.Linear(FCfinal, 4 * 3)
            )
            self.dims = [3, 4]

            # Initialize the weights/bias with identity transformation
            #self.fc_loc[2].weight.data.zero_()
            #self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # print("xs size = {}".format(xs.size()))
        if x.dim() == 5:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3] * xs.size()[4]) # This is for 3D only
        else:
            xs = xs.view(-1, xs.size()[1] * xs.size()[2] * xs.size()[3])
        theta = self.fc_loc(xs)
        # print("theta size = {}".format(theta.size()))
        theta = theta.view(-1, self.dims[0], self.dims[1])
        #print("FA: theta = {}".format(theta))

        grid = F.affine_grid(theta, self.target_grid_size, align_corners=False)
        # print("grid size = {}".format(grid.size()))
        x = F.grid_sample(x, grid, align_corners=False)

        return x, theta

    def forward(self, x):
        out, affine = self.stn(x)
        return out, affine

#############################################################################################################

class FCNConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(FCNConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size, track_running_stats=True),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size, track_running_stats=True),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size, track_running_stats=True),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class UnetGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetGatingSignal3, self).__init__()
        self.fmap_size = (4, 4, 4)

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(in_size//2, track_running_stats=True),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = self.conv1(inputs)
        outputs = outputs.view(batch_size, -1)
        outputs = self.fc1(outputs)
        return outputs


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size, track_running_stats=True), # OS
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, init_kernel=(3,3,3), init_padding =(1,1,1), init_stride=(1,1,1), is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=init_kernel, padding_size=init_padding, init_stride=init_stride)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

# OS: Deformable conv
class UnetUp3_CT_deformable(nn.Module):
    def __init__(self, in_size, out_size, init_kernel=3, init_padding=1, is_batchnorm=True):
        super(UnetUp3_CT_deformable, self).__init__()
        self.conv = UnetConv3_deformable(in_size + out_size, out_size, is_batchnorm, kernel_size=init_kernel, padding_size=init_padding)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3_deformable') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        out, _ = self.conv(torch.cat([outputs1, outputs2], 1))
        return out

# OS: My implementation of a dense upsampling block
class UnetUp3_dense_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, depth=3):
        super(UnetUp3_dense_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        self.relu = nn.ReLU(inplace=False)
        self.is_batchnorm = is_batchnorm

        # Depth of the dense block
        # -----------------------------------
        if depth < 1:
            depth = 1  # Minimal depth is 1
        self.dense_depth = depth

        # Initial convolution operation on the input data
        self.init_conv = nn.Conv3d(in_size + out_size, out_size, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        # Define further convolutions in a for loop
        self.ops_dict = {}
        for ii in range(depth - 1):
            name = "op_{}".format(ii + 1)
            self.ops_dict[name] = nn.Conv3d((ii + 1) * out_size, out_size, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)).cuda()

        self.batchnorm = nn.BatchNorm3d(out_size, track_running_stats=True)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)

        # plug the input into the dense layer
        input_to_dense = torch.cat([outputs1, outputs2], 1)

        dense = self.init_conv(input_to_dense)
        for ii in range(self.dense_depth - 1):
            name = "op_{}".format(ii + 1)

            # Batch (0) x Channels (1) x W x H x D
            outcnn = self.relu(self.ops_dict[name](dense))

            # Concatenate and relu
            if ii == self.dense_depth - 2:
                dense = self.relu(outcnn)
            else:
                dense = self.relu(torch.cat([dense, outcnn], 1))

        # Apply batch normalization if required
        if self.is_batchnorm:
            dense = self.relu(self.batchnorm(dense))

        # Output of the dense layer
        return dense

# Squeeze-and-Excitation Network
class SqEx(nn.Module):

    def __init__(self, n_features, reduction=6):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool3d(x, kernel_size=x.size()[2:5])
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        y = x * y
        return y

class UnetUp3_SqEx(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm):
        super(UnetUp3_SqEx, self).__init__()
        if is_deconv:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        concat = torch.cat([outputs1, outputs2], 1)
        gated  = self.sqex(concat)
        return self.conv(gated)

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels,  n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(n_filters,  n_filters, k_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(n_filters,  n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class SeqModelFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(SeqModelFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)

# Save nifti file to disk
def save_nifti(data, header=[], save_directory='.', savename='default_name'):
    import SimpleITK as sitk
    import os
    import numpy as np

    try:
        # If the data is a pytorch variable, send it to cpu (assume it is on gpu)
        data = data.detach().cpu().numpy()
    except:
        data = data

    # Assume the last 3 dimensions of the data are the 3d image cube
    if data.ndim == 5:
        input_arr = np.squeeze(data[0, 0, ...].cpu().numpy()).astype(np.float32)
    elif data.ndim == 4:
        input_arr = np.squeeze(data[0, ...].cpu().numpy()).astype(np.float32)
    else:
        input_arr = data.astype(np.float32)

    # Write the header and arrange data to write
    input_img = sitk.GetImageFromArray(np.transpose(input_arr, (2, 1, 0)))
    input_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])

    # Save the data as nifti
    sitk.WriteImage(input_img, os.path.join(save_directory, '{}.nii.gz'.format(savename)))