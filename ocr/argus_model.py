import torch
from torch.nn import CTCLoss

from argus.model import Model
from argus.utils import deep_to

from ocr.model import CRNN
from ocr.converter import strLabelConverter


class CRNNModel(Model):
    nn_module = {"CRNN": CRNN}
    loss = CTCLoss

    def __init__(self, params):
        super().__init__(params)
        self.alphabet = params["alphabet"]
        self.converter = strLabelConverter(params["alphabet"], device=self.device)
        self.device = torch.device(params["device"])

    def prepare_batch(self, batch, device):
        images, texts = batch["image"], batch["text"]
        output = (deep_to(images, device, non_blocking=True), texts)
        return output

    def train_step(self, batch, state) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()

        self.optimizer.zero_grad()
        images = batch["image"].to(self.device)
        # print([batch['text']])
        texts, length = self.converter.encode(batch["text"]) # USE CONVERTER TO CONVERT TEXT TO INTs and MOVE TO DEVICE TEXT
        text = torch.IntTensor(texts).to(self.device)
        length = length.to(self.device)
        preds = self.nn_module(images)
        print(torch.argmax(preds, 2)[:, 0])
        sim_preds, preds_size = self.preds_converter(preds, images.size(0))
        preds_size = preds_size.to(self.device)
        loss = self.loss(preds, text, preds_size, length)  # here ctc loss
        # print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.nn_module.parameters(), 10)
        self.optimizer.step()
        # print(sim_preds, batch['text'])
        return {
            "prediction": sim_preds,
            "target": batch['text'],
            "loss": loss.item(),
        }

    def val_step(self, batch, state) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            texts, length = self.converter.encode(
                batch["text"])  # USE CONVERTER TO CONVERT TEXT TO INTs and MOVE TO DEVICE TEXT
            text = torch.IntTensor(texts).to(self.device)
            # print(len(texts))
            length = length.to(self.device)
            preds = self.nn_module(images)
            sim_preds, preds_size = self.preds_converter(preds, images.size(0))
            print(torch.argmax(preds, 2)[:, 0])
            preds_size = preds_size.to(self.device)
            loss = self.loss(preds, text, preds_size, length)  # here ctc loss
            return {
                "prediction": sim_preds,
                "target": batch['text'],
                "loss": loss.item()
            }

    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images).to(self.device)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)

        return sim_preds, preds_size
