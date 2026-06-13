import numpy as np
import torch
import matplotlib.pyplot as plt
from src.pricers.neural_net import MLPPricer
from src.greeks.greeks import black_scholes_delta, black_scholes_vega


def compute_neural_greeks(model, S, K, T, r, sigma, device):
    """Computes Neural Delta and Vega using torch.autograd.

    Args:
        model (nn.Module): The neural network pricer.
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        device (torch.device): Torch device.

    Returns:
        tuple: A tuple containing (neural_delta, neural_vega).
    """
    model.eval()

    # Inputs must require gradient
    S_tensor = torch.tensor([S], dtype=torch.float32, requires_grad=True, device=device)
    K_tensor = torch.tensor([K], dtype=torch.float32, device=device)
    T_tensor = torch.tensor([T], dtype=torch.float32, device=device)
    r_tensor = torch.tensor([r], dtype=torch.float32, device=device)
    sigma_tensor = torch.tensor(
        [sigma], dtype=torch.float32, requires_grad=True, device=device
    )

    # Concatenate inputs. S and sigma require grads.
    inputs = torch.cat(
        [S_tensor, K_tensor, T_tensor, r_tensor, sigma_tensor]
    ).unsqueeze(0)

    # Forward pass
    price = model(inputs)

    # Compute gradients w.r.t S (Delta) and sigma (Vega)
    # create_graph=True allows higher order derivatives if needed, but we just need first order
    grads = torch.autograd.grad(
        price,
        (S_tensor, sigma_tensor),
        grad_outputs=torch.ones_like(price),
        create_graph=False,
    )

    neural_delta = grads[0].item()
    neural_vega = grads[1].item()

    return neural_delta, neural_vega


def compare_greeks(model, device):
    """Compare Neural Greeks with Black-Scholes Greeks for selected test points.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Torch device.
    """
    print("--- Comparing Neural vs Analytical Greeks ---\n")

    test_points = [
        {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "name": "ATM"},
        {"S": 120.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "name": "ITM"},
        {"S": 80.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "name": "OTM"},
    ]

    print(
        f"{'Type':<5} | {'Neural Delta':<12} | {'BS Delta':<10} | {'Neural Vega':<12} | {'BS Vega':<10}"
    )
    print("-" * 65)

    for pt in test_points:
        S, K, T, r, sigma = pt["S"], pt["K"], pt["T"], pt["r"], pt["sigma"]

        # Analytical
        bs_delta_call, _ = black_scholes_delta(S, K, T, r, sigma)
        bs_vega = black_scholes_vega(S, K, T, r, sigma)

        # Neural
        neural_delta, neural_vega = compute_neural_greeks(
            model, S, K, T, r, sigma, device
        )

        print(
            f"{pt['name']:<5} | {neural_delta:<12.4f} | {bs_delta_call:<10.4f} | {neural_vega:<12.4f} | {bs_vega:<10.4f}"
        )


def plot_delta_vs_moneyness(model, device):
    """Plot Neural Delta vs Black-Scholes Delta across moneyness.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Torch device.
    """
    print("\n--- Generating Neural Delta vs BS Delta Plot ---")
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    S_range = np.linspace(50, 150, 100)
    moneyness = S_range / K

    neural_deltas = []
    bs_deltas = []

    for S in S_range:
        nd, _ = compute_neural_greeks(model, S, K, T, r, sigma, device)
        neural_deltas.append(nd)

        bsd, _ = black_scholes_delta(S, K, T, r, sigma)
        bs_deltas.append(bsd)

    plt.figure(figsize=(10, 6))
    plt.plot(
        moneyness, neural_deltas, label="Neural Delta (orange)", color="#f97316", lw=2
    )
    plt.plot(
        moneyness,
        bs_deltas,
        label="BS Delta (blue)",
        color="#3b82f6",
        lw=2,
        linestyle="--",
    )

    plt.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7, label="ATM (S/K = 1)")
    plt.xlabel("Moneyness (S/K)")
    plt.ylabel("Delta")
    plt.title("Neural Delta vs Black-Scholes Delta across Moneyness")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("neural_vs_bs_delta.png", dpi=300)
    plt.close()
    print("Plot saved to 'neural_vs_bs_delta.png'.")


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load model
    model = MLPPricer().to(device)
    try:
        model.load_state_dict(torch.load("mlp_pricer.pth", map_location=device))
        print(f"Loaded trained model from mlp_pricer.pth to {device}")
    except FileNotFoundError:
        print("Error: mlp_pricer.pth not found. Please train the model first.")
        exit(1)

    compare_greeks(model, device)
    plot_delta_vs_moneyness(model, device)
