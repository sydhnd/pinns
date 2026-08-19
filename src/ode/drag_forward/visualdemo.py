import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt




def validate_model(model):
    # m is mass , g is gravity, k is resistence constant
    m,g,k = 10,9.8,16

    t_test = np.linspace(0, 4, 200)

    # solution of the equation of motion 2nd order ODE
    v_terminal = (m * g) / k
    y_exact = v_terminal * t_test - (m * v_terminal / k) * (1 - np.exp(-k * t_test / m))


    with torch.no_grad():
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)
        y_pred = model(t_test_tensor).numpy().flatten()

    l2_rel_error = np.linalg.norm(y_pred - y_exact) / np.linalg.norm(y_exact)
    print(f"\nFinal Relative L2 Error: {l2_rel_error * 100:.3f}%")

    plt.figure(figsize=(9, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(t_test, y_exact, 'k--', label='Analytical Exact', linewidth=2)
    plt.plot(t_test, y_pred, 'r-', label='PINN Prediction', alpha=0.7)
    plt.xlabel('Time t (s)')
    plt.ylabel('Displacement y(t) (m)')
    plt.title('Falling Object: PINN vs Analytical')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_test, np.abs(y_pred - y_exact), 'b-')
    plt.xlabel('Time t (s)')
    plt.ylabel('Absolute Error (m)')
    plt.title('Pointwise Absolute Error')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("pinn_newton_air_results.png", dpi=300)
    print("Saved comparison plot to 'pinn_newton_air_results.png'")
    plt.show()