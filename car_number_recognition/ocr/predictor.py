# HERE YOUR PREDICTOR
from argus.model import load_model
import torch
import torchvision
from ocr.converter import strLabelConverter
from ocr.transforms import *
import string
from tabulate import tabulate


class Predictor:
    def __init__(self, model_path, image_size, device="cpu"):
        self.device = device
        self.model = load_model(model_path, device=device)
        self.ocr_image_size = image_size
        h, w = image_size
        self.transform = torchvision.transforms.Compose([Scale((h,w)),
                  CentralCrop((h,w)),
               ImageNormalization(),
               ImageNormalizationMeanStd(),
               ToTensor()]) #TODO: prediction_transform
        # alphabet = " "
        alphabet = "-ABEKMHOPCTYX"
        alphabet += "".join([str(i) for i in range(10)])
        self.converter = strLabelConverter(alphabet, device=self.device)



    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images).to(self.device)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)

        return sim_preds, preds_size


    def predict(self, images):
        #TODO: check for correct input type, you can receive one image [x,y,3] or batch [b,x,y,3]
        assert (len(images.shape)==3 or len(images.shape)==4) and images.shape[-1]==3
        if len(images.shape)==3:
            images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            images = self.transform(images).unsqueeze(0).to(self.device)
        else:
            out = []
            for image in images:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                out.append(self.transform(image))
            images = torch.stack(out).to(self.device)
        pred = self.model.predict(images)
        # print(torch.argmax(pred, 2))
        text = self.preds_converter(pred, images.size(0))
        return text[0]

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import os
    import cv2
    from ocr.transforms import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Directory with images", required=True)
    parser.add_argument("--model", help="Path to model", required=True)
    parser.add_argument("--cuda", action="store_true", help="If gpu should be used")
    parser.add_argument("--show", action="store_true", help="If images should be shown")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"
    image_paths = os.listdir(args.data_dir)
    predictor = Predictor(args.model, (32, 80), device)
    out_list = []
    for filename in image_paths[:30]:
        fullpath = os.path.join(args.data_dir, filename)
        img = cv2.imread(fullpath)
        text = predictor.predict(img)
        if args.show:
            plt.imshow(img)
            plt.title(text)
            plt.show()
        out_list.append((filename, text))

    print(tabulate(out_list, headers=['Filename', 'Predicted number']))

