import torch
from torch.utils.data import TensorDataset


def create_dummy_dataset(num_samples: int, num_features: int) -> TensorDataset:
    X = torch.zeros((num_samples, num_features))
    for i in range(1, num_features + 1):
        std_dev = (10 / i**2)**0.5
        X[:, i - 1] = torch.normal(0, std_dev, size=(num_samples,))

    w1 = torch.normal(0, 1, size=(num_features,))
    w2 = torch.normal(0, 1, size=(num_features,))
    b1 = torch.normal(0, 1, size=(1,))
    b2 = torch.normal(0, 1, size=(1,))

    logit1 = torch.matmul(X, w1) + b1
    logit2 = torch.matmul(X, w2) + b2
    logits = torch.min(logit1, logit2)

    probabilities = torch.sigmoid(logits)

    labels = torch.bernoulli(probabilities)

    return TensorDataset(X, labels)
