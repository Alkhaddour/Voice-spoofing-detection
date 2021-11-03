# The following three variables hold the directory of the train, validation and test sets respectively
TRAIN_RAW_DIR = 'E:/Datasets/ID R&D/data/raw/Training_Data/'
TEST_RAW_DIR = 'E:/Datasets/ID R&D/data/raw/Testing_Data/'
# The following three variables hold the directory of the processed train, validation and test sets respectively before
# applying any scaling.
TRAIN_PROCESSED_DIR = 'E:/Datasets/ID R&D/data/processed/Training_Data'
VAL_PROCESSED_DIR = 'E:/Datasets/ID R&D/data/processed/Validation_Data'
TEST_PROCESSED_DIR = 'E:/Datasets/ID R&D/data/processed/Testing_Data'
# The following three variables hold the directory of the scaled train, validation and test sets respectively.
TRAIN_SCALED_DIR = 'E:/Datasets/ID R&D/data/scaled/Training_Data'
VAL_SCALED_DIR = 'E:/Datasets/ID R&D/data/scaled/Validation_Data'
TEST_SCALED_DIR = 'E:/Datasets/ID R&D/data/scaled/Testing_Data'
# The following three variable hold respectively the name of train, validation and test sets used in this project.
TRAIN_INDEX = 'E:/Datasets/ID R&D/data/scaled/train_index.pkl'
VAL_INDEX = 'E:/Datasets/ID R&D/data/scaled/val_index.pkl'
TEST_INDEX = 'E:/Datasets/ID R&D/data/scaled/test_index.pkl'
# The following variable contains the path of the scaler fitted using the training set.
SCALER_PATH = 'E:/Datasets/ID R&D/data/processed/standard_scaler.pkl'
# ----------------- data preprocessing hyper-params ----------------------------
VAL_PCT = 0.2
SAMPLE_RATE = 16000  # (in Hz) the expected sample rate for LibriTTS dataset
# -------------------------- MFCC parameters -----------------------------------
FRAME_SIZE = 25  # MFCC frame size (ms)
FRAME_STEP = 10  # MFCC step size  (ms)
N_FEATURES = 40  # Number of cepstral coefficients
N_FILT = 40  # Number of filters
N_FFT = int(2 * SAMPLE_RATE * FRAME_SIZE / 1000)
# -------------------------- Model hyper-parameters --------------------------
# Model params
INPUT_SIZE = 40
HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 3
LINEAR_SIZE = 512
OUTPUT_SIZE = 1
BATCH_SIZE = 64
BATCH_FIRST = True

MLP_1_HIDDEN_SIZE = 128  # dimension of the hidden layer in the model
N_EPOCHS = 60  # Number of the epochs for training the model
LR = 5e-4  # Learning rate
DROPOUT_RATE = 0  # Dropout rate used in the dropout layer after the hidden layer
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.7
PATIENCE = 20    # Number of epochs to wait before stopping the training process in case the validation loss is not
                # changing to a better value.
MODELS_DIR = './models'  # Directory to save the trained models
MODEL_NAME = 'AntiSpoof'
DEVICE = 'cuda'  # Accelerator used in training 'cuda' or 'cpu'
OUTPUT_DIR = './output'
# General variables
N_CLASSES = 2
SHOW_WARNINGS = True  # if True, the warning I create appears to output stream
SHOW_INFO_MESSAGES = True  # if True, any info message I create appears on the output stream.
PROGRESS_STEP_TRAIN = 150 # print progress every PROGRESS_STEP mini-batches
PROGRESS_STEP_VAL = 50
CLASS_CODE_MAP = {'spoof': 1, 'human': 0}  # dictionary to convert CLASS to code
CODE_CLASS_MAP = {1: 'spoof', 0: 'human'}  # dictionary to convert code to CLASS

CLASS_CODE_MAP.keys()
