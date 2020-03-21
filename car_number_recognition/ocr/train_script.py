import argparse

from torch.utils.data import DataLoader
from argus.callbacks import MonitorCheckpoint
from ocr.dataset import OcrDataset
from ocr.argus_model import CRNNModel
from config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
# from ocr.utils import regular, negative
# from ocr.transforms import get_transforms
from ocr.metrics import StringAccuracy, CER
import string
from pathlib import Path
import torch

from ocr.transforms import *
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# IF YOU USE GPU UNCOMMENT NEXT LINES:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [
    Path(CV_CONFIG.get("data_path"))
]
# CHANGE YOUR BATCH SIZE
BATCH_SIZE = 32
# 400 EPOCH SHOULD BE ENOUGH
NUM_EPOCHS = 1200

# alphabet = " "
alphabet = "-ABEKMHOPCTYX"
alphabet += "".join([str(i) for i in range(10)])

MODEL_PARAMS = {"nn_module":
                    ("CRNN", {
                        "image_height": CV_CONFIG.get("model_image_height"),
                        "number_input_channels": CV_CONFIG.get("model_image_ch"),
                        "number_class_symbols": len(alphabet),
                        "rnn_size": CV_CONFIG.get("model_rnn_size")
                    }),
                "alphabet": alphabet,
                "loss": {"reduction":"mean"},
                "optimizer": ("Adam", {"lr": 0.0001}),
                # CHANGE DEVICE IF YOU USE GPU
                "device": "cuda",
                }

if __name__ == "__main__":
    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    h, w = CV_CONFIG.get("ocr_image_size")
    train_transforms = [RandomScale((h, w)),
                        RandomRotation(p=0.6),
                        RandomBlur(p=0.3),
                        RandomCrop((h, w)),
                        RandomPad(min_scale=0.6),
                        RandomBrightness(),
                        RandomContrast(),
                        ImageNormalization(),
                        ImageNormalizationMeanStd(),
                        ToTensor()]
    val_transforms = [Scale((h, w)),
                CentralCrop((h,w)),
               ImageNormalization(),
               ImageNormalizationMeanStd(),
               ToTensor()]
    # define data path

    train_dataset = OcrDataset(DATASET_PATHS[0], transforms=train_transforms, train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_dataset = OcrDataset(DATASET_PATHS[0], transforms=val_transforms, train=False)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = CRNNModel(MODEL_PARAMS)

    callbacks = [
        MonitorCheckpoint(EXPERIMENT_DIR, monitor="val_char_error_rate", max_saves=6),
    ]

    metrics = [CER()]
    model.fit(
        train_loader,
        val_loader=val_loader,
        max_epochs=NUM_EPOCHS,
        metrics=metrics,
        callbacks=callbacks,
        metrics_on_train=True,

    )
