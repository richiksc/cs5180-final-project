# GrAdE: gradient adaptive entropy
import torch
import torch.nn as nn

### Usage
### init with your nn.Module:
### grade = GradientAdaptiveEntropy(actor_net, entropy_coef=0.01, beta=0.9)
###
### Use in entropy bonus in loss
### policy_loss = policy_loss - grade.get_bonus(entropy)

from typing import Optional

class GradientAdaptiveEntropy:
    def __init__(
        self,
        module: nn.Module,
        entropy_coef: float = 0.01,
        beta: float = 0.9,
        eps: float = 1e-8
    ) -> None:
        """
        Initialize Gradient Adaptive Entropy tracking.

        Args:
            module: The nn.Module to track gradients of
            entropy_coef: Base entropy coefficient
            beta: EMA decay rate (default: 0.9)
            eps: Small constant for numerical stability
        """
        self.module = module
        self.entropy_coef = entropy_coef
        self.beta = beta
        self.eps = eps

        # Initialize tracking variables
        self.grad_mag_ema: Optional[float] = None
        self.max_grad_mag_ema: Optional[float] = None

    def compute_grad_magnitudes(self) -> float:
        """Compute the total gradient magnitude across all parameters."""
        total_grad = 0.0
        for param in self.module.parameters():
            if param.grad is not None:
                total_grad += torch.norm(param.grad.data).item()
        return total_grad

    def get_bonus(self, current_entropy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the adaptive entropy bonus.

        Args:
            current_entropy: Current policy entropy value

        Returns:
            Scaled entropy bonus based on gradient magnitudes
        """
        # Compute current gradient magnitudes
        current_grad_mag = self.compute_grad_magnitudes()

        # Update EMA of gradient magnitudes
        if self.grad_mag_ema is None:
            self.grad_mag_ema = current_grad_mag
        else:
            self.grad_mag_ema = (self.beta * self.grad_mag_ema +
                               (1 - self.beta) * current_grad_mag)

        # Update maximum observed EMA
        if self.max_grad_mag_ema is None:
            self.max_grad_mag_ema = self.grad_mag_ema
        else:
            self.max_grad_mag_ema = max(self.max_grad_mag_ema, self.grad_mag_ema)

        # Compute ratio with numerical stability
        ema_ratio = self.grad_mag_ema / (self.max_grad_mag_ema + self.eps)

        # Return scaled entropy bonus
        return self.entropy_coef * ema_ratio * current_entropy

    def get_stats(self):
        ema_ratio = self.grad_mag_ema / (self.max_grad_mag_ema + self.eps)
        return {
            'grade.grad_mag_ema': self.grad_mag_ema,
            'grade.max_grad_mag_ema': self.max_grad_mag_ema,
            'grade.ema_ratio': ema_ratio
        }


    def reset_stats(self) -> None:
        """Reset the EMA tracking statistics."""
        self.grad_mag_ema = None
        self.max_grad_mag_ema = None



