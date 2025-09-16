import numpy as np
import pandas as pd
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_cosmology_data(n_samples=500):
    # Generate random values within specified ranges
    h0 = np.random.uniform(65, 75, n_samples)
    om_m = np.random.uniform(0.2, 0.4, n_samples)
    om_lambda = np.random.uniform(0.6, 0.8, n_samples)
    w = np.random.uniform(-1.5, -0.8, n_samples)
    om_k = np.random.uniform(-0.05, 0.05, n_samples)
    
    # Define scenarios
    scenarios = ['Big Freeze', 'Big Rip', 'Big Crunch', 'Vacuum Decay']
    scenario = [random.choice(scenarios) for _ in range(n_samples)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'Hubble_Constant_H0': h0,
        'Omega_m': om_m,
        'Omega_Lambda': om_lambda,
        'w': w,
        'Omega_k': om_k,
        'Scenario': scenario
    })
    return data

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save data
    output_file = os.path.join(output_dir, 'cosmological_parameters.csv')
    data = generate_cosmology_data()
    data.to_csv(output_file, index=False)
    
    # Print first 5 rows as a sample
    print("First 5 rows of the generated dataset:")
    print(data.head().to_string(index=False))
    print(f"\nDataset with {len(data)} rows generated and saved to: {output_file}")
