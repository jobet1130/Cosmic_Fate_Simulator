import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_monte_carlo_data(n_simulations=10000):
    # Generate random parameters
    simulation_ids = np.arange(1, n_simulations + 1)
    scenarios = np.random.choice(
        ['Big Freeze', 'Big Rip', 'Big Crunch', 'Vacuum Decay'],
        size=n_simulations,
        p=[0.3, 0.2, 0.3, 0.2]  # Adjust probabilities as needed
    )
    
    # Generate random cosmological parameters
    h0 = np.random.uniform(65, 75, n_simulations)
    omega_m = np.random.uniform(0.2, 0.4, n_simulations)
    omega_lambda = np.random.uniform(0.6, 0.8, n_simulations)
    w = np.random.uniform(-1.5, -0.8, n_simulations)
    
    # Generate time to end based on scenario
    time_to_end = np.zeros(n_simulations)
    
    for i, scenario in enumerate(scenarios):
        if scenario == 'Big Freeze':
            time_to_end[i] = np.random.uniform(100, 1000)
        elif scenario == 'Big Rip':
            time_to_end[i] = np.random.uniform(20, 80)
        elif scenario == 'Big Crunch':
            time_to_end[i] = np.random.uniform(50, 150)
        else:  # Vacuum Decay
            time_to_end[i] = np.random.uniform(0, 200)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Simulation_ID': simulation_ids,
        'Scenario': scenarios,
        'H0': np.round(h0, 6),
        'Omega_m': np.round(omega_m, 6),
        'Omega_Lambda': np.round(omega_lambda, 6),
        'w': np.round(w, 6),
        'Time_to_End_Gyr': np.round(time_to_end, 6)
    })
    
    return df

if __name__ == "__main__":
    # Generate the data
    df = generate_monte_carlo_data()
    
    # Ensure output directory exists
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/raw/monte_carlo_times.csv'
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"Monte Carlo time-to-end data saved to {output_file}")
    print(f"Total rows: {len(df)}")
    print("Sample data:")
    print(df.head())
