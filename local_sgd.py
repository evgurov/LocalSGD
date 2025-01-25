import typing as tp
import copy
import wandb

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader


class LocalSGD:
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
        self.loss_fn = loss_fn

        self.main_model = copy.deepcopy(model)
        self.models: list[nn.Module] = [
            copy.deepcopy(model) for _ in range(num_workers)
        ]
        self.optimizers: list[torch.optim.Optimizer] = [
            SGD(model.parameters(), lr = lr) for model in self.models
        ]
        self.loader = DataLoader(train_dataset, batch_size=num_workers * K)

    # One "parallel" training round
    # =============================================================================================

    def _training_round(
        self,
        total_round_X: torch.Tensor,
        total_round_y: torch.Tensor,
    ) -> None:

        for worker_idx in range(self.num_workers):
            worker_X = total_round_X[
                (self.K * worker_idx) : (self.K * (worker_idx + 1))
            ]
            worker_y = total_round_y[
                (self.K * worker_idx) : (self.K * (worker_idx + 1))
            ]
            model = self.models[worker_idx]
            optimizer = self.optimizers[worker_idx]
            for local_step in range(self.K):
                sample, label = (
                    worker_X[local_step].unsqueeze(0), 
                    worker_y[local_step].unsqueeze(0),
                )
                output = model(sample).squeeze(1)
                loss = self.loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Average models
    # =============================================================================================

    def _average_models(self):
        with torch.no_grad():
            for main_model_param, param_group in zip(
                self.main_model.parameters(),
                zip(*[model.parameters() for model in self.models])
            ):
                main_model_param.data.copy_(
                    sum([param.data for param in param_group]) / len(self.models)
                )
                for param in param_group:
                    param.data.copy_(main_model_param.data)

    # "Evaluate" main model on train dataset
    # =============================================================================================
            
    def evaluate_model(self) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.loader:
                output = self.main_model(X).squeeze(1)
                loss = self.loss_fn(output, y)
                total_loss += loss
                
            total_loss /= len(self.loader)
            
        return total_loss

    # Main train loop
    # =============================================================================================

    def train(self, num_epochs: int) -> None:
        wandb.init(
            name = f"LocalSGD: M={self.num_workers}, K={self.K}"
        )

        for epoch_num in range(num_epochs):
            for total_round_X, total_round_y in tqdm(
                self.loader, desc=f"Epoch {epoch_num + 1}"
            ):
                current_loss = self.evaluate_model()
                wandb.log({"loss": current_loss})

                self._training_round(total_round_X, total_round_y)
                self._average_models()
                
        current_loss = self.evaluate_model()
        wandb.log({"loss": current_loss})
                
        wandb.finish()
