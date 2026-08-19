import torch
import torch.nn as nn
import torch.optim as optim


# The real decay rate of SODIUM24 is 0.046
REAL_K_SODIUM24 = 0.046336

# Inverse PINN example for finding k in a 1st-order radio decay ODE
class InverseDecayPINN(nn.Module):
    def __init__(self):
        super().__init__()

        #one uknown parameter k 
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # initial with education guess  (in decay model k < 1)
        self.k = nn.Parameter(torch.tensor([0.6], requires_grad=True))

    def forward(self, t):
        return self.net(t)


model = InverseDecayPINN()
# Inital value
u0 = 10

# calculated observed value at t = 30 mins, t = 60 mins t = 2 hours t = 3 hours
t_obs = torch.tensor([[0.0], [1.0], [1.5], [2.0], [3.0],[14.0]], requires_grad=False)
u_obs = u0 * torch.exp(REAL_K_SODIUM24 * t_obs)


t_domain = torch.linspace(0, 14, 50, requires_grad=True).view(-1, 1)

optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(3000):
    optimizer.zero_grad()

    # loss data
    loss_data = torch.mean((model(t_obs) - u_obs)**2)

    # loss residual
    u_dom = model(t_domain)
    du_dt = torch.autograd.grad(u_dom, t_domain, torch.ones_like(u_dom), create_graph=True)[0]
    residual = du_dt - model.k * u_dom
    loss_physics = torch.mean(residual**2)

    # Total loss
    loss = loss_data + loss_physics
    loss.backward()
    optimizer.step() 

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Learned k: {model.k.item():.4f} (Target: 0.0463)")        


