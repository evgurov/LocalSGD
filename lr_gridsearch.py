import typing as tp

from torch import nn
from torch.utils.data import Dataset


def lr_gridsearch(
    algorithm: type,
    model: nn.Module,
    dataset: Dataset,
    loss_fn: tp.Callable,
    num_workers: int,
    K: int,
    num_epochs: int,
    lr_grid: tp.Iterable[float]
) -> float:

    best_loss = float('inf')
    best_lr = None
    for lr in lr_grid:
        algo_instance = algorithm(model, dataset, loss_fn, num_workers, K, lr)
        algo_instance.train(num_epochs = num_epochs)
        loss = algo_instance.evaluate_model()
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            
    return best_lr