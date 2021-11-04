import pickle

import torch

from RNN_trainer import predict_batch
from config import LSTM_NUM_LAYERS, HIDDEN_SIZE, FRAME_SIZE, FRAME_STEP, N_FEATURES, N_FILT, N_FFT, SCALER_PATH, \
    BATCH_FIRST, INPUT_SIZE, LINEAR_SIZE, OUTPUT_SIZE, CODE_CLASS_MAP, MODEL_NAME, MODELS_DIR
from data_processing.feature_extractor import mfcc_feature_extractor
from models import AntiSpoofingRNN
from utilities.basic_utils import get_accelerator, make_valid_path
from utilities.disply_utils import info
from utilities.model_utils import Metrics, ModelManager


def test(model, test_loader, pred_threshold=0.5, device=get_accelerator('cuda')):
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


# if __name__ == '__main__':
#     model = AntiSpoofingRNN(INPUT_SIZE, HIDDEN_SIZE, LSTM_NUM_LAYERS, LINEAR_SIZE, OUTPUT_SIZE)
#     model_manager = ModelManager(MODEL_NAME, make_valid_path(MODELS_DIR, is_dir=True, exist_ok=True))
#     model, _ = model_manager.load_checkpoint(MODEL_NAME + '.pkl', model)
#
#     audio_path = 'E:/Datasets/ID R&D/data/raw/Training_Data/human/human_00000.wav'
#     # Load scaler
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#
#     out = test_sample(model, audio_path, scaler, pred_threshold=0.2630763351917267)
#     info(f"Sample classified as {CODE_CLASS_MAP[out]}")

# if __name__ == '__main__':
#     info("Testing model")
#     model = AntiSpoofingRNN(INPUT_SIZE, HIDDEN_SIZE, LSTM_NUM_LAYERS, LINEAR_SIZE, OUTPUT_SIZE)
#     model_manager = ModelManager(MODEL_NAME, make_valid_path(MODELS_DIR, is_dir=True, exist_ok=True))
#     model, _ = model_manager.load_checkpoint(MODEL_NAME + '.pkl', model)
#
#     info("Validating performance on train data")
#     train_dataset = ReplySpoofDataset(TRAIN_INDEX)
#     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn_pad)
#     files, y_true, y_pred, y_prob, threshold = test(model, train_loader, pred_threshold='auto',
#                                                     device=get_accelerator('cuda'))
#     export_incorrect_samples_to_csv(files, y_pred, y_true, os.path.join(OUTPUT_DIR, 'train_incorrect.csv'))
#     info(f"Classification threshold = {threshold}")  # threshold = 0.23641443252563477
#     info("Calculating train metrics")
#     train_metrics, train_metrics_str = Metrics.calculate_metrics(y_true, y_pred, y_prob)
#     info(f"Train accuracy = {train_metrics['Accuracy']:0.4f}")
#
#     info("Validating performance on validation data")
#     val_dataset = ReplySpoofDataset(VAL_INDEX)
#     val_loader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn_pad)
#     files, y_true, y_pred, y_prob, threshold = test(model, val_loader, pred_threshold=threshold,
#                                                     device=get_accelerator('cuda'))
#     export_incorrect_samples_to_csv(files, y_pred, y_true, os.path.join(OUTPUT_DIR, 'val_incorrect.csv'))
#     info("Calculating validation metrics")
#     val_metrics, val_metrics_str = Metrics.calculate_metrics(y_true, y_pred, y_prob)
#     info(f"Validation accuracy = {val_metrics['Accuracy']:0.4f}")
#
#     with open(os.path.join(OUTPUT_DIR, 'metrics.csv'), 'w') as f:
#         train_metrics['#'] = 'Train'
#         val_metrics['#'] = 'Validation'
#         w = csv.DictWriter(f, sorted(val_metrics.keys()))
#         w.writeheader()
#         w.writerow(train_metrics)
#         w.writerow(val_metrics)
#
#     info("Testing model done!")
