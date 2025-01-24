import typing as tp
import copy
import wandb

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader


class MinibatchSGD:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        loss_fn: tp.Callable,
        num_workers: int = 1,
        K: int = 1,
        lr: float = 1e-3,
    ) -> None:

        self.num_workers = num_workers
        self.K = K
        self.model = copy.deepcopy(model)
        self.loss_fn = loss_fn

        self.optimizer = SGD(self.model.parameters(), lr = lr)
        self.loader = DataLoader(train_dataset, batch_size=num_workers * K)

    # One "parallel" training round
    # =============================================================================================

    def _training_round(
        self,
        total_round_X: torch.Tensor,
        total_round_y: torch.Tensor,
    ) -> None:

        output = self.model(total_round_X).squeeze(1)
        loss = self.loss_fn(output, total_round_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # "Evaluate" model on train dataset
    # =============================================================================================

    def _evaluate_model(self) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.loader:
                output = self.model(X).squeeze(1)
                loss = self.loss_fn(output, y)
                total_loss += loss
                
            total_loss /= len(self.loader)

        return total_loss

    # Main train loop
    # =============================================================================================

    def train(
        self,
        num_epochs: int,
    ) -> None:
        wandb.init(
            name = f"MinibatchSGD: M={self.num_workers}, K={self.K}"
        )

        for epoch_num in range(num_epochs):
            for total_round_X, total_round_y in tqdm(
                self.loader, desc=f"Epoch {epoch_num + 1}"
            ):
                self._training_round(total_round_X, total_round_y)
                current_loss = self._evaluate_model()
                
                wandb.log({"loss": current_loss})
                
        wandb.finish()
