# Abstract level model definition
# Returns the model class for specified network type
import os


class ModelOpts:
    def __init__(self):
        self.gpu_ids = [0]
        self.isTrain = True
        self.continue_train = False
        self.which_epoch = int(0)
        self.save_dir = './checkpoints/default'
        self.model_type = 'unet'
        self.input_nc = 1
        self.output_nc = 4
        self.lr_rate = 1e-12
        self.l2_reg_weight = 0.0
        self.feature_scale = 4
        self.tensor_dim = '2D'
        self.path_pre_trained_model = None
        self.criterion = 'cross_entropy'
        self.type = 'seg'

        # Attention
        self.nonlocal_mode = 'concatenation'
        self.attention_dsample = (2,2,2)

        # Attention Classifier
        self.aggregation_mode = 'concatenation'

        # Data dimensions
        self.dataDims = [8, 1, 96, 192, 112]

        # Energy loss
        self.energy_lambda = 1e-6
        self.energy_temperature = 1
        self.energy_threshold = -8

        self.optim = "adam"

    def initialise(self, json_opts):
        opts = json_opts

        self.raw = json_opts
        self.gpu_ids = opts.gpu_ids
        self.isTrain = opts.isTrain
        self.save_dir = os.path.join(opts.checkpoints_dir, opts.experiment_name)
        self.model_type = opts.model_type
        self.input_nc = opts.input_nc
        self.output_nc = opts.output_nc
        self.continue_train = opts.continue_train
        self.which_epoch = opts.which_epoch

        if hasattr(opts, 'type'): self.type = opts.type
        if hasattr(opts, 'l2_reg_weight'): self.l2_reg_weight = opts.l2_reg_weight
        if hasattr(opts, 'lr_rate'):       self.lr_rate = opts.lr_rate
        if hasattr(opts, 'feature_scale'): self.feature_scale = opts.feature_scale
        if hasattr(opts, 'tensor_dim'):    self.tensor_dim = opts.tensor_dim

        if hasattr(opts, 'path_pre_trained_model'): self.path_pre_trained_model = opts.path_pre_trained_model
        if hasattr(opts, 'criterion'):              self.criterion = opts.criterion

        if hasattr(opts, 'nonlocal_mode'): self.nonlocal_mode = opts.nonlocal_mode
        if hasattr(opts, 'attention_dsample'): self.attention_dsample = opts.attention_dsample
        # Classifier
        if hasattr(opts, 'aggregation_mode'): self.aggregation_mode = opts.aggregation_mode

        if hasattr(opts, 'optim'): self.optim = opts.optim

        # OS: may addition to the Tversky - DICE loss
        if hasattr(opts, 'alpha'): self.alpha = opts.alpha
        if hasattr(opts, 'beta'): self.beta = opts.beta
        if hasattr(opts, 'lam'): self.lam = opts.lam
        if hasattr(opts, 'hausdorff'): self.hausdorff = opts.hausdorff

        # Energy loss
        if hasattr(opts, 'energy_lambda'): self.energy_lambda = opts.energy_lambda
        if hasattr(opts, 'energy_threshold'): self.energy_threshold = opts.energy_threshold
        if hasattr(opts, 'energy_temperature'): self.energy_temperature = opts.energy_temperature

# OS: Added the batchsize as well - needed for the STN based network
def get_model(json_opts, dataDims, rank=0):

    # Neural Network Model Initialisation
    model = None
    model_opts = ModelOpts()
    model_opts.initialise(json_opts)
    model_opts.dataDims = dataDims
    model_opts.rank = rank # For multi-processing only

    # Print the model type
    if rank == 0:
        print('\nInitialising model {}'.format(model_opts.model_type))

    model_type = model_opts.type
    if model_type == 'seg':
        # Multimodal (i.e. only axial, but T1, T2, FA, etc.)
        from .feedforward_seg_model import FeedForwardSegmentation
        model = FeedForwardSegmentation()

    elif model_type == 'seg_mult':
        # Multi projection (i.e. axial and coronal) - also includes multimodal
        from .feedforward_seg_model import FeedForwardSegmentation_multproj
        model = FeedForwardSegmentation_multproj()

    elif model_type == 'seg_unreg':
        # Multi projection (i.e. axial and coronal) - also includes multimodal
        from .feedforward_seg_model import FeedForwardSegmentation_unreg
        model = FeedForwardSegmentation_unreg()

    elif model_type == 'reg':
        # T2 axial - coronal registration
        from .feedforward_seg_model import FeedForward_registration
        model = FeedForward_registration()

    elif model_type == 'classifier':
        # Return the model type
        from .feedforward_classifier import FeedForwardClassifier
        model = FeedForwardClassifier()

    elif model_type == 'aggregated_classifier':
        # Return the model type
        from .aggregated_classifier import AggregatedClassifier
        model = AggregatedClassifier()


    # Initialise the created model
    model.initialize(model_opts)
    if model_opts.rank == 0:
        print("Model [%s] is created" % (model.name()))

    return model
