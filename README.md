# Physics-Informed Neural Networks (PINNs)
 
A PINN builds physics directly into a neural network. Instead of learning purely from data, it's trained to satisfy a differential equation, so the physics itself (along with any known initial or boundary conditions) becomes part of the loss function.
 
## Two Ways to Use PINNs
 
### Forward Problems
 
Given governing equations, physical parameters, initial conditions (ICs), and boundary conditions (BCs), the network is trained to approximate the system state $u(t)$ or $u(x, t)$ over the domain.
 
**Example (2nd-Order ODE with Linear Drag):**
 
$$\frac{d^2 y}{dt^2} + \frac{b}{m}\frac{dy}{dt} = g$$
 
### Inverse Problems
 
If some parameters are unknown, say a rate constant, a damping coefficient, or a diffusivity, the network can use a handful of measurements to reconstruct the system's behavior and work out those missing parameters at the same time.
 
**Example (first-order ODE: radioactive decay):**
 
$$\frac{du}{dt} = -k u$$
 
### PDEs
 
The PINN approach is not limited to ODEs — it extends naturally to partial differential equations (PDEs) as well, but the calculation becomes much more complicated. I am not going to discuss it in detail here.
 
**Example Heat / Diffusion Equation (Parabolic PDE):**
 
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$
 
## Theory: Why It Works
 
A PINN trains like any other network, by minimizing a total loss function:
 
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{residual}} + \lambda_{\text{IC}}\mathcal{L}_{\text{IC}} + \lambda_{\text{BC}}\mathcal{L}_{\text{BC}}$$
 
with a gradient-based optimizer like Adam or L-BFGS.
 
### Initial and Boundary Conditions
 
#### Initial Condition
 
if condition $u(0)= a $ and $u'(0)=b$ are given and $\hat{u}(t)$ is the model-predicted function.
 
$$\mathcal{L}_{\text{IC}_1} = (\hat{u}(0) - a)^2$$
 
$$\mathcal{L}_{\text{IC}_2} = (\frac{d\hat{u}}{dt}(0) - b)^2$$
 
$$\mathcal{L}_{\text{IC}} = \mathcal{L}_{\text{IC}_1} + \mathcal{L}_{\text{IC}_2}$$
 

### Boundary Condition
 
if boundary $u(x_0) = a$ and $u(x_1) = b$ (here $x_0$ and $x_1$ denote the two domain endpoints, not time)
 
$$\mathcal{L}_{\text{BC}_0} = (\hat{u}(x_0) - a)^2$$
 
$$\mathcal{L}_{\text{BC}_1} = (\hat{u}(x_1) - b)^2$$
 
$$\mathcal{L}_{\text{BC}} = \mathcal{L}_{\text{BC}_0} + \mathcal{L}_{\text{BC}_1}$$
 
These same $\mathcal{L}_{\text{IC}}$ and $\mathcal{L}_{\text{BC}}$ terms are used in both the forward and inverse loss functions below.

### Forward Problem Loss
 
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{residual}} + \lambda_{\text{IC}}\mathcal{L}_{\text{IC}} + \lambda_{\text{BC}}\mathcal{L}_{\text{BC}}$$
 
$\mathcal{L}_{\text{residual}}$ (Physics Loss): Evaluates how well $\hat{u}$ satisfies the differential equation residual at sampled collocation points.
 
$\mathcal{L}_{\text{residual}}$ is the physics loss. It checks how badly the network's output breaks the governing equation, evaluated at $N$ points scattered across the domain (usually called collocation points). $\mathcal{L}_{\text{IC}}$ pulls the network toward the known starting state at $t = 0$, and $\mathcal{L}_{\text{BC}}$ does the same job at the domain's boundaries.
 
#### Example: First-Order Decay System
 
$$\frac{du}{dt} = -ku, \qquad k = 0.5\text{ min}^{-1},\quad u(0) = 10$$
 
| Time ($t$, min) | Analytical Formula | Expected $u(t)$ |
| :---: | :---: | :---: |
| $0$ | $10 \cdot e^{0}$ | $10.0000$ |
| $1$ | $10 \cdot e^{-0.5}$ | $6.0653$ |
| $2$ | $10 \cdot e^{-1.0}$ | $3.6788$ |
| $3$ | $10 \cdot e^{-1.5}$ | $2.2313$ |
 
$$\frac{du}{dt} = -ku \quad\Longrightarrow\quad \frac{du}{dt} + ku = 0$$
 
when $t = 0$:
 
$$\mathcal{R}(t_0) = \frac{d\hat{u}}{dt}(0) + 0.5\, \hat{u}(0)$$
 
 
The residual formula will become
 
$$\mathcal{L}_{\text{residual}} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{d\hat{u}}{dt}(t_i) + k \hat{u}(t_i) \right)^2$$
 


### Inverse Problem Loss
 
When you're solving for unknown parameters, real measurements help pin the system down. The goal is to minimize the cost function
 
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{residual}} + \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{IC}}\mathcal{L}_{\text{IC}} + \lambda_{\text{BC}}\mathcal{L}_{\text{BC}}$$
 
#### Example: Parameter Discovery in a Damped Harmonic Oscillator
 
$$\frac{d^2 u}{dt^2} + 2\gamma \frac{du}{dt} + \omega_0^2 u = 0, \qquad u(0) = 1, \quad \frac{du}{dt}(0) = 0$$
 
Here I want to recover the damping coefficient $\gamma$ and natural frequency $\omega_0$ using $M$ measured data points $u_{\text{obs}}(t_i)$.
 
The physics residual loss:
 
$$\mathcal{R}(t_i) = \frac{d^2\hat{u}}{dt^2}(t_i) + 2\hat{\gamma} \frac{d\hat{u}}{dt}(t_i) + \hat{\omega}_0^2 \hat{u}(t_i)$$
 
$$\mathcal{L}_{\text{residual}} = \frac{1}{N} \sum_{i=1}^N \mathcal{R}(t_i)^2$$
 
And the data loss, comparing predictions against the actual sensor readings:
 
$$\mathcal{L}_{\text{data}} = \frac{1}{M} \sum_{i=1}^M \left(\hat{u}(t_i) - u_{\text{obs}}(t_i)\right)^2$$
 

### Partial Differential Equations
 
The math for PDEs gets a lot more involved (check the implementation for the full details). With an ODE, we're just approximating a single variable changing over time, and the network only needs one input. For a PDE, we're approximating $u(x, t)$, so the network needs more than one input feeding into a few more hidden layers
 
```python
self.net = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)
```
 
For example, consider the 1D heat diffusion equation:
 
$$\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0 \quad \text{for } x \in [-1, 1], \; t \in [0, 1]$$
 
We build a neural network to approximate $u(x,t)$:
 
```python
self.net = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)
```
 
**First derivatives**

Since u(x,t) now depends on two variables, we compute:

$$\frac{\partial \hat{u}}{\partial t}(x, t), \qquad \frac{\partial \hat{u}}{\partial x}(x, t)$$

**Second spatial derivative**

$$\frac{\partial^2 \hat{u}}{\partial x^2} = \frac{\partial}{\partial x} \left( \frac{\partial \hat{u}}{\partial x} \right)$$

 
You may wonder why there's no $\frac{\partial^2 \hat{u}}{\partial t^2}$ term. This is because heat has no inertia with respect to time in this equation. Compare this to Newton's second law:
 
$$F = m \cdot a = m \frac{d^2x}{dt^2}$$
 
where inertia comes from the mass term.
 
**Loss function**
 
The loss function must also account for two different variables, but the overall calculation is similar to the ODE version.
 
For example, the physics-informed residual loss at $x_i$ and $t_i$ can be calculated as:
 
$$\mathcal{R}(x_i, t_i; \hat{\alpha}) = \frac{\partial \hat{u}}{\partial t}(x_i, t_i) - \hat{\alpha} \frac{\partial^2 \hat{u}}{\partial x^2}(x_i, t_i)$$
 
## Repository Overview
 
| Module | Task Type | Description | Link |
| :--- | :--- | :--- | :--- |
| **ODE Inverse** | Parameter Discovery | Infers rate constant $k$ from exponential decay data |[](src/dummy/model.js) |
| **Second-Order ODE** | Forward Problem | Simulates a falling mass under gravity and linear drag | | 
| **PDE Inverse** | Parameter Discovery | Estimates the thermal diffusivity $\alpha$ in the 1D heat equation | | 


---

# discussion

### Activation function

unlike standard deep learning test `ReLU` fails in PINNs.
 
`ReLU` is piecewise linear
 
$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases} \quad \Longrightarrow \quad \frac{d^2}{dx^2}\text{ReLU}(x) = 0 \quad (\forall x \neq 0)$$
 
Its second-order spatial and temporal derivatives:
$\frac{\partial^2 u}{\partial x^2}$
$\frac{\partial^2 u}{\partial t^2}$
evaluate to zero almost everywherel t^2}$, evaluate to zero almost everywhere.
 
### Python autograd
 
In standard machine learning, PyTorch throws away the computational graph by default immediately after loss.backward() to save GPU memory. In PINNS are differentiating the same network output so create_graph=True keeps the computation graph alive so it can be differentiated again, which means the intermediate activations stay in memory (VRAM if the model is on GPU) instead of being freed after the backward pass.

```python
# 1st Temporal Derivative: du/dt
u_t = torch.autograd.grad(
    outputs=u,
    inputs=t,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
    retain_graph=True
)[0]
 
# 2nd Temporal Derivative: d^2u/dt^2
u_tt = torch.autograd.grad(
    outputs=u_t,
    inputs=t,
    grad_outputs=torch.ones_like(u_t),
    create_graph=True,
    retain_graph=True
)[0]
```
 