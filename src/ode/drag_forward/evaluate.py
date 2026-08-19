import torch
import numpy as np
import torch.nn as nn

# Inference with model to predict positoin, velcovity, acceleration at any time
def calculate_projection(model:nn.Module,t_values: np.ndarray):
    """
    Computes Position y(t), Velocity v(t), and Acceleration a(t)
    directly using the neural network and PyTorch Autograd.
    """
    # Prepare input tensor with gradient tracking enabled
    t_tensor = torch.tensor(t_values, dtype=torch.float32).view(-1, 1).requires_grad_(True)

    # Postion
    y_pred = model(t_tensor)

    # Velocity
    v_pred = torch.autograd.grad(
        outputs=y_pred,
        inputs=t_tensor,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # Acceleration
    a_pred = torch.autograd.grad(
        outputs=v_pred,
        inputs=t_tensor,
        grad_outputs=torch.ones_like(v_pred),
        create_graph=False 
    )[0]

    return {
        "time": t_values,
        "position": y_pred.detach().cpu().numpy().flatten(),
        "velocity": v_pred.detach().cpu().numpy().flatten(),
        "acceleration": a_pred.detach().cpu().numpy().flatten(),
    }