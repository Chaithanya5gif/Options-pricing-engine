import time
from src.pricers.heston import heston_mc_price
from scipy.optimize import minimize

def calibrate_heston_for_option(S0, K, T, r, market_price, seed):
    def objective(params):
        kappa, theta, sigma_v, rho, v0 = params
        if kappa < 0 or theta < 0 or sigma_v < 0 or v0 < 0 or rho < -1 or rho > 1:
            return 1e6
        
        res = heston_mc_price(S0, K, T, r, v0=v0, kappa=kappa, theta=theta, 
                              sigma_v=sigma_v, rho=rho, n_paths=2000, n_steps=10, seed=seed)
        return (res["price"] - market_price)**2
        
    initial_guess = [2.0, 0.04, 0.3, -0.7, 0.04]
    result = minimize(objective, initial_guess, method='Nelder-Mead', options={'maxiter': 50})
    return result.x

t0 = time.time()
print(calibrate_heston_for_option(100.0, 100.0, 1.0, 0.05, 10.45, 42))
print("Time:", time.time() - t0)
