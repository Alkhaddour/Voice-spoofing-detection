# This file contains the functions used to test or use model. The first function is test() which is used when the data
# used to test the model when a dataloader is available. The function expects the loader to provide data with its
# labels.

import torch
from RNN_trainer import predict_batch
from data_processing.feature_extractor import mfcc_feature_extractor
from utilities.basic_utils import get_accelerator
from utilities.model_utils import Metrics
from tqdm import tqdm
import time
from config import FRAME_SIZE, FRAME_STEP, N_FEATURES, N_FILT, N_FFT, BATCH_FIRST, CODE_CLASS_MAP


def test(model, test_loader, pred_threshold='auto', device=get_accelerator('cuda')):
    """
    Test model using data from data loader
    :param model: model to be tested
    :param test_loader: Test Loader (pytorch dataloader)
    :param pred_threshold: The threshold used to identify to which class belongs the sample being tested. If
    pred_threshold == 'auto', then the best threshold is determined using precision-recall curve, we used the value
    which gives the highest F1-score. If pred_threshold == 'eer' then we search for threshold which gives the best EER.
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
    bar = tqdm(enumerate(test_loader))
    time.sleep(1.0)
    for test_batch in bar:
        batch_id, (files, batch, labels) = test_batch
        bar.set_description(f"Processing batch #{batch_id:4d}/{len(test_loader)}")
        labels = labels.reshape(-1).float().to(device)
        outputs = predict_batch(model, batch, device)

        labels_all = labels_all + list(labels.detach().cpu().numpy().astype(int))
        probs_all = probs_all + list(outputs.detach().cpu().numpy())
        files_all = files_all + list(files)
    if pred_threshold == 'auto':
        pred_threshold, _ = Metrics.get_best_prec_recall_threshold(labels_all, probs_all)
    elif pred_threshold == 'eer':
        pred_threshold, _ = Metrics.get_best_eer_threshold(labels_all, probs_all)
    else:
        try:
            pred_threshold = float(pred_threshold)
        except Exception:
            assert False, "Expected a float threshold or 'auto' or 'eer'"
    outputs_all = [0 if x < pred_threshold else 1 for x in probs_all]
    return files_all, labels_all, outputs_all, probs_all, pred_threshold


def test_sample(model, audio_path, scaler, return_types=['code'], pred_threshold=0.5, device=get_accelerator('cuda')):
    """
    Predict class of an audio file
    :param model: Model used for prediction
    :param audio_path: Input wave file to classify
    :param scaler: Scaler used to scale data
    :param return_types: List or one string of the following return types:
                        'code' to return class index,
                        'class_name' to return a string representing the class
                        'score' to return testing score. Score represents the probability that an audio was recorded by
                                a live voice
    :param pred_threshold: Threshold used to determine class from probability.
    :param device: Accelerator used (GPU or CPU)
    :return: A single return_type if return_types is a string, otherwise a tuple of the required return types
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

    probs = predict_batch(model, features, device)
    output = 0 if probs < pred_threshold else 1

    if type(return_types) == str:
        return_types = [return_types]
    else:
        assert type(return_types) == list, "return_type must be a string or list"

    outputs = []
    for return_type in return_types:
        if return_type == 'code':
            outputs.append(output)
        elif return_type == 'class_name':
            outputs.append(CODE_CLASS_MAP[output])
        elif return_type == 'score':
            outputs.append(probs.item())
        else:
            raise Exception("Undefined return type, allowed values are ['code', 'class_name', 'score']")
    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)
