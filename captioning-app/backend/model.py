import torch
import pickle
import numpy as np
import os
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
import torch

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, inputs):
        if inputs.dim() == 4:
            with torch.no_grad():
                features = self.resnet(inputs)
                features = self.pool(features).squeeze(-1).squeeze(-1)
            features = self.linear(features)
            features = self.bn(features)
            return features
        elif inputs.dim() == 5:
            B, T, C, H, W = inputs.size()
            inputs = inputs.view(B * T, C, H, W)
            with torch.no_grad():
                features = self.resnet(inputs)
                features = self.pool(features).squeeze(-1).squeeze(-1)
            features = self.linear(features)
            features = self.bn(features)
            features = features.view(B, T, -1)
            features = features.mean(dim=1)
            return features
        else:
            raise ValueError("Unsupported input shape for EncoderCNN")

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)
        hiddens, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(hiddens)
        return outputs
