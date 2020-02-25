from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torch



training_data = [
    [3.0, 1.5],
    [2.0, 1.0],
    [4.0, 1.5],
    [3.0, 1.0],
    [3.5, 0.5],
    [2.0, 0.5],
    [5.5, 1.0],
    [1.0, 1.0]
]

labels = [
        [1],
        [0],
        [1],
        [0],
        [1],
        [0],
        [1],
        [0]
]



training_data = torch.FloatTensor(training_data)
labels = torch.FloatTensor(labels)


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.sigmoid(out)

        return out




cost_function = nn.MSELoss()


model = FeedforwardNeuralNetModel(2, 1)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

costs = []

for epoch in range(5000):


    for i, dat in enumerate(training_data):

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        label = labels[i]

        # Forward pass to get output/logits
        prediction = model.forward(dat)

        loss = cost_function(prediction, label)

        costs.append(loss)

        loss.backward()
        optimizer.step()



mistery_flower = [
    [4.5, 1]
]

mistery_flower = torch.FloatTensor(mistery_flower)
pred = model.forward(mistery_flower)


print(pred)

plt.plot(costs)
plt.show()
