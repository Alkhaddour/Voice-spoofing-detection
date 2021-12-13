# ---------------------------------------- Directories and file paths for data ----------------------------------------
# The following two variables hold the directory of the train and test sets respectively
# Each directory contains 2 folders, the name of the folder is the class name (human, spoof).
# For test set there are no sub directories as the test set has no public labels
TRAIN_RAW_DIR = 'E:/Datasets/ID R&D/data/raw/Training_Data/'
TEST_RAW_DIR = 'E:/Datasets/ID R&D/data/raw/Testing_Data/'
# The following three variables hold the directory of the processed train, validation and test sets respectively before
# applying any scaling. the processed files are numpy files containing the features extracted from the audio files
TRAIN_PROCESSED_DIR = 'E:/Datasets/ID R&D/data/processed/Training_Data'
VAL_PROCESSED_DIR = 'E:/Datasets/ID R&D/data/processed/Validation_Data'
TEST_PROCESSED_DIR = 'E:/Datasets/ID R&D/data/processed/Testing_Data'
# The following three variables hold the directory of the scaled train, validation and test sets respectively. The files
# are numpy files, each represent the features extracted from an audio file and scaled according to statistics obtained
# from the train set
TRAIN_SCALED_DIR = 'E:/Datasets/ID R&D/data/scaled/Training_Data'
VAL_SCALED_DIR = 'E:/Datasets/ID R&D/data/scaled/Validation_Data'
TEST_SCALED_DIR = 'E:/Datasets/ID R&D/data/scaled/Testing_Data'
# The following three variable representing the index of the train, val and test sets respectively. An index contains
# a pair of file_path and a label. The file path represents link to features extracted from one audio file.
TRAIN_INDEX = 'E:/Datasets/ID R&D/data/scaled/train_index.pkl'
VAL_INDEX = 'E:/Datasets/ID R&D/data/scaled/val_index.pkl'
TEST_INDEX = 'E:/Datasets/ID R&D/data/scaled/test_index.pkl'
# The following variable contains the path of the scaler fitted using the training set.
SCALER_PATH = 'E:/Datasets/ID R&D/data/processed/standard_scaler.pkl'

# ------------------------------------------ Data preprocessing hyper-params ------------------------------------------
VAL_PCT = 0.2           # Percentage of train set used for validation
SAMPLE_RATE = 16000     # (in Hz) the expected sample rate for audio files

# -------------------------------------------------- MFCC parameters --------------------------------------------------
FRAME_SIZE = 25  # MFCC frame size (ms)
FRAME_STEP = 10  # MFCC step size  (ms)
N_FEATURES = 40  # Number of cepstral coefficients
N_FILT = 40      # Number of filters
N_FFT = 512

# ------------------------------------------------- Model parameters --------------------------------------------------
# Model params
INPUT_SIZE = 40         # Model input size
HIDDEN_SIZE = 128       # LSTM hidden size
LSTM_NUM_LAYERS = 3     # Number of LSTM layers
LINEAR_SIZE = 512       # Size of projection size
OUTPUT_SIZE = 1         # Model output size
BATCH_SIZE = 64
BATCH_FIRST = True
N_EPOCHS = 60               # Number of the epochs for training the model
LR = 5e-4                   # Initial Learning rate
SCHEDULER_STEP_SIZE = 10    #
SCHEDULER_GAMMA = 0.7       #
PATIENCE = 20               # Number of epochs to wait before stopping the training process in case the validation loss
                            # is not changing to a better value.
MODELS_DIR = './models'     # Directory to save the trained models
MODEL_NAME = 'AntiSpoof'    # Model name
DEVICE = 'cuda'             # Accelerator used in training 'cuda' or 'cpu'
OUTPUT_DIR = './output'     # Directory of any output generated beside model files

# ------------------------------------------------- General variables -------------------------------------------------
SHOW_WARNINGS = True        # if True, the warning I create appears to output stream
SHOW_INFO_MESSAGES = True   # if True, any info message I create appears on the output stream.
PROGRESS_STEP_TRAIN = 150   # print progress every PROGRESS_STEP mini-batches (training)
PROGRESS_STEP_VAL = 50      # print progress every PROGRESS_STEP mini-batches (validation)
CLASS_CODE_MAP = {'spoof': 0, 'human': 1}  # dictionary to convert CLASS to code
CODE_CLASS_MAP = {0: 'spoof', 1: 'human'}  # dictionary to convert code to CLASS

