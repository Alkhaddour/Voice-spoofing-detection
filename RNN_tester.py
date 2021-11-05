# This file contains the functions used to test or use model. The first function is test() which is used when the data
# used to test the model when a dataloader is available. The function expects the loader to provide data with its
# labels.

import torch
from RNN_trainer import predict_batch
from data_processing.feature_extractor import mfcc_feature_extractor
from utilities.basic_utils import get_accelerator
from utilities.disply_utils import info
from utilities.model_utils import Metrics
from config import FRAME_SIZE, FRAME_STEP, N_FEATURES, N_FILT, N_FFT, BATCH_FIRST, CODE_CLASS_MAP


def test(model, test_loader, pred_threshold=0.5, device=get_accelerator('cuda')):
    """
    Test model using data from data loader
    :param model: model to be tested
    :param test_loader: Test Loader (pytorch dataloader)
    :param pred_threshold: The threshold used to identify to which class belongs the sample being tested. If
    pred_threshold == 'auto', then the best threshold is determined using precision-recall curve, we used the value
    which gives the highest F1-score.
    :param device: Accelerator used (GPU or CPU)
    :return:    files_all: list of files used to test model
                labels_all: list of true labels
                outputs_all: list of predicted labels
                probs_all: list of probabilities, if probability is less than pred_threshold then the the sample is of
                           class 0, otherwise it belongs to class 1
                pred_threshold: the threshold used to determine class

    """
    model = model.to(device)
    labels_all = []
    probs_all = []
    files_all = []
    for batch_id, (files, batch, labels) in enumerate(test_loader):
        if (batch_id % 50) == 0:
            info(f"Processing batch [{batch_id + 1}/{len(test_loader)}]")
        labels = labels.reshape(-1).float().to(device)
        outputs = predict_batch(model, batch, device)

        labels_all = labels_all + list(labels.detach().cpu().numpy().astype(int))
        probs_all = probs_all + list(outputs.detach().cpu().numpy())
        files_all = files_all + list(files)
    if pred_threshold == 'auto':
        pred_threshold, _ = Metrics.get_best_prec_recall_threshold(labels_all, probs_all)

    outputs_all = [0 if x < pred_threshold else 1 for x in probs_all]
    return files_all, labels_all, outputs_all, probs_all, pred_threshold


def test_sample(model, audio_path, scaler, return_code=True, pred_threshold=0.5, device=get_accelerator('cuda')):
    """
    Predict class of an audio file
    :param model: Model used for prediction
    :param audio_path: Input wave file to classify
    :param scaler: Scaler used to scale data
    :param return_code: True then return class index, otherwise return a string representing the class
    :param pred_threshold: Threshold used to determine class from probability.
    :param device: Accelerator used (GPU or CPU)
    :return: Class of audio file
    """
    feature_extractor = mfcc_feature_extractor(frame_size=FRAME_SIZE, frame_step=FRAME_STEP, numcep=N_FEATURES,
                                               nfilt=N_FILT, nfft=N_FFT)
    features = feature_extractor.process_audio(audio_path)
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32).to(device)
    if BATCH_FIRST:
        features = features.unsqueeze(0)
    else:
        features = features.unsqueeze(1)

    output = predict_batch(model, features, device)
    output = 0 if output < pred_threshold else 1
    if return_code:
        return output
    else:
        return CODE_CLASS_MAP[output]

