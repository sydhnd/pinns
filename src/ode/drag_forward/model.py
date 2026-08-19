import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import visualdemo

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)


# m is mass , g is gravity, k is resistence constant
m,g,k = 10,9.8,16

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(3000):
        optimizer.zero_grad()
        t_domain = torch.linspace(0, 4, 200, requires_grad=True).view(-1, 1)
        y = model(t_domain)

        dy_dt = torch.autograd.grad(
        y, t_domain,
        grad_outputs=torch.ones_like(y),
        create_graph=True
        )[0]

        # 2nd order derivative d^2y/dt^2
        d2y_dt2 = torch.autograd.grad(
        dy_dt, t_domain,
        grad_outputs=torch.ones_like(dy_dt),
        create_graph=True
        )[0]        

        residual = d2y_dt2 + k/m *  dy_dt - g
        loss_residual = torch.mean(residual**2)

        # Initial condition point t = 0 
        t_0 = torch.tensor([[0.0]], requires_grad=True)
        y_0 = model(t_0)

        dy_dt_0 = torch.autograd.grad(
        y_0, t_0,grad_outputs=torch.ones_like(y_0),
        create_graph=True
        )[0]

        # loss weight multipler to stablize the solution
        loss_weight_multipler = 100.0
        loss_ic = torch.mean((y_0 - 0.0)**2 + (dy_dt_0 - 0.0)**2)
        total_loss = loss_residual + loss_weight_multipler * loss_ic
        total_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Residual Loss: {loss_residual.item():.6f} | IC Loss: {loss_ic.item():.6f}")


# compare the visual graph 
model.eval()

# demo the different between solution and predicted model
visualdemo.validate_model(model)

# calualte the projection for following time point
query_times = np.array([0.5, 1.25, 2.0, 3.75])

predictions = evaluate.calculate_projection(model,query_times)
for i in range(len(predictions["time"])):
    print(f"t = {predictions['time'][i]:.2f}s | "
          f"Position y(t) = {predictions['position'][i]:.3f} m | "
          f"Velocity v(t) = {predictions['velocity'][i]:.3f} m/s | "
          f"Acceleration a(t) = {predictions['acceleration'][i]:.3f} m/s²")