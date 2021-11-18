# A rede neural é um modulo que consiste de outros modulos (layers)
# Essa estrutura emaranhada permite facilmente a criação e gestão de 
# arquiteturas complexas.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Definimos a rede neural com as subclasses nn.module, e inicializamos os layers
# da rede em __init__. Cada subclasse do nn.module implementa sua operação no
# dado imputado.


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Aqui criamos a instancia da NeuralNetwork e a movemos para o device, e 
# printamos sua estrutura

model = NeuralNetwork().to(device)
print(model)

# Para utilizar o modelo, passamos os dados do input. 

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print(f"Predicted class: {y_pred}")