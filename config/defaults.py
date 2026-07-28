from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '1'
# Name of backbone
_C.MODEL.NAME = 'MDReID'
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH_T = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
_C.MODEL.PROMPT = False # From MambaPro
_C.MODEL.ADAPTER = False # From MambaPro
_C.MODEL.FROZEN = False # whether to freeze the backbone
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with the contact feature
_C.MODEL.DIRECT = 1

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.GLOBAL_LOCAL = False # Whether to use the local information in PIFE for MDReID
_C.MODEL.HEAD = 12 # Number of heads in the ATMoE

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False  # We do not use this parameter
_C.MODEL.ADD_SHARE = False
_C.MODEL.ADD_CLOSS = False
_C.MODEL.ADD_ELOSS = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('RGBNT201')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14  # This may be affected by the order of data reading
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16  # You can adjust it to 8 to save memory while the batch_size need to be 64 to ensure the number of ID

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.009
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
_C.MODEL.NO_MARGIN = True
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128  # You can adjust it to 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Pattern of test augmentation
_C.TEST.MISS = 'None'

# ---------------------------------------------------------------------------- #
# Exemplar-free lifelong multi-modal ReID
# ---------------------------------------------------------------------------- #
_C.LIFELONG = CN()
_C.LIFELONG.ENABLED = False
_C.LIFELONG.TRACK = "A"
_C.LIFELONG.ORDER = "grouped"
_C.LIFELONG.ADAPTER_RANK = 64
_C.LIFELONG.ADAPTER_DROPOUT = 0.0
# Options: mean, latest, zero. "mean" initializes a new same-modality adapter
# from the uniform average of all historical task adapters.
_C.LIFELONG.ADAPTER_INIT = "mean"
_C.LIFELONG.ROUTING = "auto"
_C.LIFELONG.EVAL_SCENARIOS = ("RNT", "R", "N", "T")
# Development monitor only. It never selects or saves a best checkpoint.
_C.LIFELONG.PERIODIC_EVAL_PERIOD = 5
_C.LIFELONG.PERIODIC_EVAL_SCENARIOS = ("RNT",)
_C.LIFELONG.PERIODIC_EVAL_ROUTING = "oracle"
_C.LIFELONG.WM_MANIFEST = ""
_C.LIFELONG.AMP = True
_C.LIFELONG.GRAD_CLIP = 5.0
_C.LIFELONG.CONSISTENCY_LOSS_WEIGHT = 0.1
_C.LIFELONG.ROUTER = CN()
# Options:
# - "gaussian_fingerprint": fixed single-Gaussian domain fingerprints.
# - "task_key": learnable multi-modal task keys.
# - "legacy": raw consistency/classifier-entropy heuristic.
_C.LIFELONG.ROUTER.METHOD = "task_key"
_C.LIFELONG.ROUTER.CATEGORY_AWARE = True
# Gaussian fingerprint settings. Statistics are fitted on the current task's
# deterministic training view and then frozen; no image or sample feature is
# retained. Diagonal variance shrinkage improves stability in 512 dimensions.
_C.LIFELONG.ROUTER.GAUSSIAN_NORMALIZE_FEATURES = True
_C.LIFELONG.ROUTER.GAUSSIAN_VARIANCE_SHRINKAGE = 0.1
_C.LIFELONG.ROUTER.GAUSSIAN_VARIANCE_FLOOR = 1e-6
_C.LIFELONG.ROUTER.GAUSSIAN_RELATIVE_VARIANCE_FLOOR = 0.05
_C.LIFELONG.ROUTER.GAUSSIAN_CALIBRATION_FLOOR = 0.05
_C.LIFELONG.ROUTER.CONSISTENCY_WEIGHT = 1.0
_C.LIFELONG.ROUTER.CONFIDENCE_WEIGHT = 0.1
_C.LIFELONG.ROUTER.KEYS_PER_MODALITY = 4
_C.LIFELONG.ROUTER.KEY_TEMPERATURE = 0.07
_C.LIFELONG.ROUTER.KEY_WEIGHT = 1.0
# Calibrate every task/modality score by its own current-train median and IQR
# before comparing task banks. This removes raw cosine-score scale bias.
_C.LIFELONG.ROUTER.TASK_KEY_CALIBRATION = True
_C.LIFELONG.ROUTER.TASK_KEY_CALIBRATION_FLOOR = 0.01
# Seed K keys from diverse adapter-free features in a few deterministic
# current-task batches instead of relying on unrelated random directions.
_C.LIFELONG.ROUTER.TASK_KEY_FEATURE_INITIALIZATION = True
_C.LIFELONG.ROUTER.TASK_KEY_INITIALIZATION_BATCHES = 4
_C.LIFELONG.ROUTER.AUX_CONSISTENCY_WEIGHT = 0.0
_C.LIFELONG.ROUTER.AUX_CONFIDENCE_WEIGHT = 0.0
_C.LIFELONG.ROUTER.LOSS_WEIGHT = 1.0
_C.LIFELONG.ROUTER.POSITIVE_WEIGHT = 1.0
_C.LIFELONG.ROUTER.MARGIN_WEIGHT = 1.0
_C.LIFELONG.ROUTER.SEPARATION_WEIGHT = 0.1
_C.LIFELONG.ROUTER.DIVERSITY_WEIGHT = 0.05
_C.LIFELONG.ROUTER.MARGIN = 0.2
_C.LIFELONG.ROUTER.SEPARATION_MARGIN = 0.2
_C.LIFELONG.ROUTER.DIVERSITY_MARGIN = 0.2
_C.LIFELONG.ROUTER.KEY_LR_MULTIPLIER = 10.0

# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./test"
