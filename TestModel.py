import torch

# Pulled from pytorch tutorial: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
class TestModel(torch.nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()

        self.linear1 = torch.nn.Linear(2048*3, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 40)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1, -1);
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x