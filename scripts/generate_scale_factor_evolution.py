import numpy as np
import pandas as pd
from math import exp, sinh, cosh, tanh

def big_freeze_scale_factor(t, H0=70):
    """Exponential growth for Big Freeze scenario"""
    return np.exp(H0 * t / 100)  # Slower exponential growth

def big_rip_scale_factor(t, H0=70):
    """Rapidly diverging scale factor for Big Rip scenario"""
    t_rip = 200  # Time of the Big Rip
    return 1 / (1 - (t/t_rip)**2)

def big_crunch_scale_factor(t, t_crunch=150):
    """Scale factor that grows then collapses to 0 for Big Crunch"""
    return np.sin(np.pi * t / (2 * t_crunch))

def vacuum_decay_scale_factor(t, H0=70):
    """Exponential decay to a constant value for Vacuum Decay"""
    return 1 + 0.5 * (1 - np.exp(-H0 * t / 500))

def generate_evolution_data():
    # Time array from 0 to 200 Gyr in steps of 0.5 Gyr
    time = np.arange(0, 200.25, 0.5)
    
    # Initialize lists to store data
    data = []
    
    # Generate data for each scenario
    scenarios = {
        'Big Freeze': big_freeze_scale_factor,
        'Big Rip': big_rip_scale_factor,
        'Big Crunch': big_crunch_scale_factor,
        'Vacuum Decay': vacuum_decay_scale_factor
    }
    
    for scenario, scale_func in scenarios.items():
        for t in time:
            a = scale_func(t)
            # Simple Hubble parameter proportional to the derivative of the scale factor
            # Add some noise to make it look more realistic
            h = 70 * (1 + 0.1 * np.random.normal()) * (scale_func(t + 0.1) - a) / (0.1 * a)
            
            # For Big Crunch, handle the collapse phase
            if scenario == 'Big Crunch' and t > 150:
                a = max(0, 2 - t/150)  # Linear collapse to 0
                h = -70  # Negative Hubble parameter for collapsing universe
            
            data.append([t, a, h, scenario])
    
    return pd.DataFrame(data, columns=['Time_Gyr', 'Scale_Factor_a', 'Hubble_Parameter_H', 'Scenario'])

if __name__ == "__main__":
    # Generate the data
    df = generate_evolution_data()
    
    # Ensure output directory exists
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/raw/scale_factor_evolution.csv'
    df.to_csv(output_file, index=False)
    print(f"Scale factor evolution data saved to {output_file}")
    print("First few rows:")
    print(df.head())
