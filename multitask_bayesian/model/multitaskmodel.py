import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    def forward(self, x):
        weight_epsilon = torch.randn(self.out_features, self.in_features)
        bias_epsilon = torch.randn(self.out_features)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon
        return F.linear(x, weight, bias)

class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))

    def forward(self, x):
        weight_epsilon = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        bias_epsilon = torch.randn(self.out_channels)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)

class MultitaskBayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim, domain_dim):
        super(MultitaskBayesianNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.domain_dim = domain_dim

        self.conv1 = BayesianConv2d(1, 60, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=1)
        self.conv2 = BayesianConv2d(60, 120, kernel_size=3, stride=1, padding=1)
        self.conv3 = BayesianConv2d(120, 200, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=1)
        self.linear1 = BayesianLinear(35200, 100)
        self.linear2 = BayesianLinear(100, 70)
        self.class_linear = BayesianLinear(70, self.output_dim)
        self.domain_linear = BayesianLinear(70, self.domain_dim)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, d):


        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)



        # Separate heads for classification and domain prediction
        y1 = self.class_linear(x)
        y2 = self.domain_linear(x)

        # Apply softmax to output probabilities
        y1 = self.softmax(y1)
        y2 = self.softmax(y2)

        y = torch.cat((y1, y2), dim=-1)
        return y