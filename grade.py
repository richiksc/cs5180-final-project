# GrAdE: gradient adaptive entropy
import torch
import torch.nn as nn

### Usage
### init with your nn.Module:
### grade = GradientAdaptiveEntropy(actor_net, entropy_coef=0.01, beta=0.9)
###
### Use in entropy bonus in loss
### policy_loss = policy_loss - grade.get_bonus(entropy)

class GradientAdaptiveEntropy:
    def __init__(self, module: nn.Module, entropy_coef=0.01, beta=0.9) -> None:
        """
        :param module the nn.Module you would like to track gradients of
        :param entropy_coef,
        """
        self.module = module
        self.max_grad_mag_ema = None
        self.grad_mag_ema = None

        self.entropy_coef = entropy_coef
        self.beta = beta # EMA rate

    def compute_grad_magnitudes(self):
        """Compute the total gradient magnitude for a model's gradients."""
        total_grad = 0.0
        for param in self.module.parameters():
            if param.grad is not None:
                total_grad += torch.norm(param.grad).item()
        return total_grad

    def get_bonus(self, current_entropy):
        if self.grad_mag_ema is None:
            self.grad_mag_ema = current_entropy
        else:
            self.grad_mag_ema = self.beta * self.grad_mag_ema + (1 - self.beta) * self.compute_grad_magnitudes()

        if self.max_grad_mag_ema is None:
            self.max_grad_mag_ema = self.grad_mag_ema
        else:
            self.max_grad_mag_ema = max(self.max_grad_mag_ema, self.grad_mag_ema)

        ema_ratio = self.grad_mag_ema / self.max_grad_mag_ema

        return self.entropy_coef * ema_ratio * current_entropy




