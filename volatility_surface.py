import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from black_scholes import implied_vol

def plot_3d_volatility_surface():
    print("Fetching options data for 3D volatility surface...")
    ticker = yf.Ticker("SPY")
    
    expirations = ticker.options
    if not expirations:
        print("No options data available.")
        return
        
    # Pick 4 near-term expiry dates to get a nice surface
    target_expiries = expirations[1:5]
    print(f"Using expirations: {target_expiries}")
    
    S = ticker.history(period="1d")["Close"].iloc[-1]
    r = 0.05
    today = datetime.date.today()
    
    strikes_all = []
    ttm_all = []
    iv_all = []
    
    for expiry in target_expiries:
        try:
            chain = ticker.option_chain(expiry)
            calls = chain.calls
            
            # Calculate time to expiration T
            expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
            days_to_expiry = (expiry_date - today).days
            if days_to_expiry <= 0:
                continue
            T = days_to_expiry / 365.0
            
            # Filter strikes near money (e.g. 80% to 120% of spot)
            calls = calls[(calls['strike'] > S * 0.85) & (calls['strike'] < S * 1.15)].copy()
            
            for _, row in calls.iterrows():
                K = row['strike']
                market_price = row['lastPrice']
                
                if market_price <= 0:
                    continue
                    
                iv = implied_vol(market_price, S, K, T, r, option_type="call")
                if iv is not None and 0 < iv < 2.0: # Filter extremes
                    strikes_all.append(K)
                    ttm_all.append(T)
                    iv_all.append(iv)
        except Exception as e:
            print(f"Failed for expiry {expiry}: {e}")
            
    if not iv_all:
        print("Not enough data to plot surface.")
        return
        
    print(f"Gathered {len(iv_all)} data points. Generating 3D plot...")
    
    # Convert to numpy arrays for plotting
    x = np.array(strikes_all)
    y = np.array(ttm_all)
    z = np.array(iv_all)
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate unstructured data into grid
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Also scatter the real data points lightly on top
    ax.scatter(x, y, z, color='k', s=5, alpha=0.3)
    
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Expiry (Years)')
    ax.set_zlabel('Implied Volatility (\u03C3)')
    ax.set_title('3D Volatility Surface (SPY)')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
    
    # Adjust viewing angle for best tweet-ability
    ax.view_init(elev=30, azim=230)
    
    plt.savefig("volatility_surface_3d.png", dpi=300, bbox_inches='tight')
    print("3D volatility surface saved as 'volatility_surface_3d.png'.")

if __name__ == "__main__":
    plot_3d_volatility_surface()
