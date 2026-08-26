import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# The real alpha  value in equation
REAL_ALPHA_VALUE  = 0.2

# Inverse PINN example to finding the alpha Parabolic PDE
class InverseHeatDiffusionPINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # education guess for inital alpha value
        self.alpha_learned = nn.Parameter(torch.tensor([0.8], requires_grad=True))

    def forward(self, x, t):  
        inputs = torch.cat([x, t], dim=1)      
        return self.net(inputs)



# observed value from experiment
observed_x = torch.tensor([[-0.5], [0.5], [-0.5], [0.5]], dtype=torch.float32)
observed_t = torch.tensor([[0.2],  [0.2], [0.5],  [0.5]], dtype=torch.float32)
observed_u = torch.tensor([[0.6738], [-0.6738], [0.3727], [-0.3727]], dtype=torch.float32)

model = InverseHeatDiffusionPINN()
optimizer = optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(3000):
    optimizer.zero_grad()

    N_colloc = 1000
    x_colloc = (torch.rand(N_colloc, 1) * 2.0 - 1.0).requires_grad_(True)
    t_colloc = torch.rand(N_colloc, 1).requires_grad_(True)
    u_colloc = model(x_colloc, t_colloc)

    du_dt = torch.autograd.grad(
        outputs=u_colloc,
        inputs=t_colloc,
        grad_outputs=torch.ones_like(u_colloc),
        create_graph=True
    )[0]

    du_dx = torch.autograd.grad(
    outputs=u_colloc,
    inputs=x_colloc,
    grad_outputs=torch.ones_like(u_colloc),
    create_graph=True
    )[0]

    d2u_dx2 = torch.autograd.grad(
    outputs=du_dx,
    inputs=x_colloc,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True
    )[0]

    residual_pde = du_dt - model.alpha_learned * d2u_dx2
    loss_pde = torch.mean(residual_pde ** 2)


    # calculate loss_ic
    N_ic = 200
    x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1)
    t_ic = torch.zeros(N_ic, 1)

    # target from calculation
    u_ic_target = -torch.sin(np.pi * x_ic)
    u_ic_pred = model(x_ic, t_ic)

    loss_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)
    loss_weight_multipler = 100.0

    u_sensor_pred = model(observed_x, observed_t)
    loss_data = torch.mean((u_sensor_pred - observed_u) ** 2)
    total_loss = loss_pde + loss_weight_multipler * loss_ic + 200  * loss_data
    total_loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | PDE Loss: {loss_pde.item():.6f} | "
            f"Data Loss: {loss_data():.6f} | "
            f"Learned Alpha: {model.alpha_learned.item():.4f} (Target: {REAL_ALPHA_VALUE})")


