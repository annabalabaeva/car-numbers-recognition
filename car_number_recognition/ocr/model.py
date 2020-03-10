#https://github.com/courao/ocr.pytorch/blob/master/recognize/crnn.py
import torch.nn as nn
from dpipe.layers import PostActivation2d


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


def convblock(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bn=nn.BatchNorm2d,
        activation=nn.ReLU,
        pooling=None
):
    conv = [
        PostActivation2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            batch_norm_module=bn,
            activation_module=activation,
        )
    ]

    if pooling is not None:
        conv.append(pooling)

    return conv


class CRNN(nn.Module):
    def __init__(
            self,
            image_height,
            number_input_channels,
            number_class_symbols,
            rnn_size
    ):
        '''
        :param image_height: As far as h == 1, image height must be equal 16
        :param number_input_channels: 3 for color image and 1 for gray scale
        :param number_class_symbols: Length of alphabet
        :param rnn_size: time length of rnn layer, 64|128|256 and so on
        '''
        super(CRNN, self).__init__()
        assert image_height % 16 == 0, "image_height has to be a multiple of 16"

        ks = [3, 3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        nm = [number_input_channels, 32, 32, 64, 64, 64, 64, 64, 128, 128]

        bn_layers = [
            None,
            None,
            nn.BatchNorm2d,
            None,
            nn.BatchNorm2d,
            None,
            nn.BatchNorm2d,
            None,
            nn.BatchNorm2d,
        ]

        poolings = [
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            None,
            None,
            nn.MaxPool2d((2, 2), (2, 1), (1, 1)),
            None,
            nn.MaxPool2d((2, 2), (2, 1), (1, 1)),
            None,
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

        ]


        relus = [nn.ReLU] * len(ks)

        cnn = []
        for i in range(len(ks)):
            cnn.extend(
                convblock(
                    in_channels=nm[i],
                    out_channels=nm[i + 1],
                    kernel_size=ks[i],
                    stride=ss[i],
                    padding=ps[i],
                    bn=bn_layers[i],
                    activation=relus[i],
                    pooling=poolings[i]
                )
            )

        self.cnn = nn.Sequential(*cnn)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(128, rnn_size, number_class_symbols)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        output = nn.functional.log_softmax(output, dim=2)

        return output
