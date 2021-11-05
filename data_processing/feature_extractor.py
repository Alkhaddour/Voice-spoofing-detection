"""
Summary:
This file defines mfcc_feature_extractor class which is used to extract mfcc feature from audio signal. The class
provides the possibility to extract MFCC features from single wav file, and also to process a directory of wav files
with saving the extracted features of each file to an output directory

The file also contains a driver function extract_features() which extract features from train and test sets and save
results to other directory, this function uses the parameters defined in the config.py file.
"""

from python_speech_features import mfcc
from utilities.basic_utils import warn, make_valid_path
from datetime import datetime
from config import *
import scipy.io.wavfile as wav
import numpy as np
import os

from utilities.disply_utils import info


class mfcc_feature_extractor:
    def __init__(self, frame_size=20, frame_step=10, numcep=13, nfilt=26, nfft=512, preemph=0, ceplifter=0, verbose=1):

        self.frame_size = frame_size  # in milliseconds
        self.frame_step = frame_step  # in milliseconds
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft
        self.preemph = preemph
        self.ceplifter = ceplifter
        self.verbose = verbose

    @staticmethod
    def _ms_to_sample(ms, sample_rate):
        """
        finds how many samples in the ms milliseconds
        :param ms:
        :param sample_rate: Sampling rate
        """
        return int(ms * sample_rate / 1000)

    @staticmethod
    def _signal_duration(self, signal, sample_rate, unit='m'):
        """
        Finds out signal duration in minutes or seconds.
        :param signal: the signal represented as list of samples
        :param sample_rate: Sample rate
        :param unit: 'm' for minutes and 's' for seconds.
        :return: signal duration
        """
        if unit == 'm':
            return len(signal) / sample_rate / 60
        elif unit == 's':
            return len(signal) / sample_rate
        else:
            raise ValueError(f'Unknown duration unit ({unit})')

    def extract_mfcc_features(self, signal, sample_rate):
        """
        Extract the MFCC feature from a signal
        :param signal: The signal to process
        :param sample_rate: the sampling rate
        :return:
        """
        mfcc_frame_size = float(self.frame_size / 1000)  # seconds
        mfcc_frame_step = float(self.frame_step / 1000)  # seconds
        mfcc_feat = mfcc(signal=signal, samplerate=sample_rate,
                         winlen=mfcc_frame_size, winstep=mfcc_frame_step,
                         numcep=self.numcep, nfilt=self.nfilt, nfft=self.nfft,
                         preemph=self.preemph, ceplifter=self.ceplifter)
        return mfcc_feat

    def process_data(self, input_dir, output_dir):
        """
        This function extracts features from each utterance and save it to disk
        :param input_dir: directory containing utterances (each of them is .wav file)
        :param output_dir: directory to save features extracted of utterances as numpy array (replace .wav file
                           extension with .npy)
        """
        for filename in os.listdir(input_dir):
            info(f'[{datetime.now()}] -- Processing file {filename} ...')
            if filename.endswith('.wav') is False:
                continue

            wavfile = os.path.join(input_dir, filename)
            mfcc_features = self.process_audio(wavfile)
            if mfcc_features is None:
                continue  # some error in processing file

            outfile = os.path.join(output_dir, f'{filename[:-4]}.npy')
            np.save(outfile, mfcc_features)

    def process_audio(self, audio_path):
        """
        This function takes a path to a wave file then extracts and returns MFCC features from it
        :param audio_path: path to a wave file
        :return: MFCC extracted from this file
        """
        sample_rate, signal = wav.read(audio_path)
        # check that the audio is sampled with the predefined value
        if sample_rate != SAMPLE_RATE:
            warn(f'[{audio_path}] has unexpected sample rate {sample_rate}, expected {SAMPLE_RATE}')
            return None
        mfcc_features = self.extract_mfcc_features(signal, sample_rate)
        return mfcc_features


# Driver part for this file, can be used to extract features from train and test sets using the default params in the
# config file
def extract_features():
    feature_extractor = mfcc_feature_extractor(frame_size=FRAME_SIZE, frame_step=FRAME_STEP, numcep=N_FEATURES,
                                               nfilt=N_FILT, nfft=N_FFT)
    # training data has two classes, each in separate folder
    for class_name in os.listdir(TRAIN_RAW_DIR):
        input_dir = os.path.join(TRAIN_RAW_DIR, class_name)
        output_dir = os.path.join(TRAIN_PROCESSED_DIR,class_name)

        feature_extractor.process_data(input_dir=input_dir, output_dir=make_valid_path(output_dir, is_dir=True))

    feature_extractor.process_data(input_dir=TEST_RAW_DIR,
                                   output_dir=make_valid_path(TEST_PROCESSED_DIR, is_dir=True))

