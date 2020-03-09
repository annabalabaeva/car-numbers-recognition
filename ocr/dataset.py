import cv2
import os
import re

from torch.utils.data import Dataset
import torchvision


class OcrDataset(Dataset):
    def __init__(self, data_path, target_path=None, transforms=None, train=False):
        images_fullpath, target = self._get_correct_data(data_path, train)
        self.data = images_fullpath
        self.target = target
        self.transforms = None
        if transforms is not None:
            self.transforms = torchvision.transforms.Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        # img_src = np.copy(img)
        t = self.target[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        # return img_src, img, t
        return {"image": img,
                "text": t}

    def _get_correct_data(self, data_path, train):
        list_filenames = sorted(os.listdir(data_path))
        if train:
            list_filenames = list_filenames[:int(len(list_filenames)*0.8)]
        else:
            list_filenames = list_filenames[int(len(list_filenames) * 0.8):]
        # pattern = re.compile("[A-Z][0-9][0-9][0-9][A-Z][A-Z] [0-9][0-9][0-9]?.bmp")
        # list_train_images = [filename for filename in list_filenames if pattern.match(filename)]
        train_fullpath = [os.path.join(data_path, filename) for filename in list_filenames]
        # from shutil import copyfile
        #
        # for n, path in zip(range(1000, len(list_train_images)), list_train_images):
        #     copyfile(os.path.join(data_path, path), os.path.join("/data/tips_tricks_data/data/good", str(n)+'_'+path))
        target = [filename[5:-4].replace(" ", "") for filename in list_filenames]
        return train_fullpath, target



def measure_time(dataset):
    import time
    start = time.time()
    for i in range(len(dataset)):
        data = dataset[i]
    end = time.time()
    return end - start


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from ocr.transforms import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Directory with images", required=True)
    parser.add_argument("--speedtest", action="store_true", help="Test time both for torchvision and custom transforms.")
    args = parser.parse_args()
    if args.speedtest:
        # (Custom transforms) time per one image: 0.0008089731743864691
        # (Torchvision transforms) time per one image: 0.001031878523504619
        dataset_custom_transforms = OcrDataset(args.data_dir,
                                               transforms=[RandomFlip(),
                                                           RandomCrop((100, 200)),
                                                           RandomBrightness(),
                                                           RandomContrast(),
                                                           ImageNormalization(),
                                                           ImageNormalizationMeanStd(),
                                                           ToTensor()])
        t = measure_time(dataset_custom_transforms)
        print("(Custom transforms) time per one image:", (t / len(dataset_custom_transforms)))
        dataset_torchvision_transforms = OcrDataset(args.data_dir,
                                                    transforms=[
                                                        torchvision.transforms.ToPILImage(),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                        torchvision.transforms.RandomCrop((100, 200), pad_if_needed=True),
                                                        torchvision.transforms.ColorJitter(brightness=0.5, contrast=2.0),
                                                        ImageNormalization(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
                                                    ])
        t = measure_time(dataset_torchvision_transforms)
        print("(Torchvision transforms) time per one image:", (t / len(dataset_torchvision_transforms)))
    else:
        dataset = OcrDataset(args.data_dir, transforms=[Scale((int(1.1*32), int(1.1*80))),
                  # RandomFlip(),
                  RandomCrop((32,80)),
               RandomBrightness(),
               RandomContrast(),
                                                        ])
        # 3110
        print("Dataset length:", len(dataset))
        # for i, (src_image, image, target) in enumerate(dataset):
        for i, (src_image, image, target) in enumerate(dataset):
            plt.subplot(211)
            plt.imshow(src_image, cmap="gray")
            plt.subplot(212)
            plt.imshow(image, cmap="gray")
            plt.title(target)
            plt.show()
            if i > 10:
                break
