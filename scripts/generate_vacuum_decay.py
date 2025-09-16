import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
np.random.seed(42)

def generate_vacuum_decay_data():
    # Parameters
    n_simulations = 100
    time_steps = np.arange(0, 50.5, 0.5)  # 0 to 50 Gyr in 0.5 Gyr steps
    universe_size = 2000  # -1000 to 1000 in both X and Y
    bubble_growth_rate = 0.5  # Mpc/Gyr
    nucleation_rate = 0.05  # probability of new bubble per time step
    
    all_data = []
    
    for sim_id in range(1, n_simulations + 1):
        # Initialize simulation state
        bubbles = []  # list of (birth_time, x, y)
        universe_fraction_remaining = 1.0
        
        for t in time_steps:
            # Check if new bubble nucleates
            if np.random.random() < nucleation_rate:
                # Generate random position for new bubble
                x = np.random.uniform(-1000, 1000)
                y = np.random.uniform(-1000, 1000)
                bubbles.append((t, x, y))
            
            # Calculate bubble radii at current time
            bubble_data = []
            for bubble_id, (birth_time, x, y) in enumerate(bubbles, 1):
                # Calculate current radius (grows linearly with time since birth)
                age = t - birth_time
                if age >= 0:  # Only process bubbles that have been born
                    radius = max(0, age * bubble_growth_rate)
                    bubble_data.append((sim_id, t, bubble_id, x, y, radius))
            
            # Calculate fraction of universe remaining (simplified model)
            if bubble_data:
                # Simple model: remaining fraction decreases with number of bubbles
                # and their total area (capped at 1.0)
                total_bubble_area = sum(np.pi * (r**2) for _, _, _, _, _, r in bubble_data)
                universe_area = universe_size**2
                universe_fraction_remaining = max(0, 1 - min(1, total_bubble_area / universe_area))
            
            # Add bubble data for this time step
            for sim_id, time, b_id, x, y, r in bubble_data:
                all_data.append([
                    sim_id,
                    round(time, 1),  # Time_Gyr
                    b_id,            # Bubble_ID
                    round(x, 2),     # Bubble_Center_X
                    round(y, 2),     # Bubble_Center_Y
                    round(r, 2),     # Bubble_Radius
                    round(universe_fraction_remaining, 6)  # Universe_Fraction_Remaining
                ])
    
    return pd.DataFrame(all_data, columns=[
        'Simulation_ID', 'Time_Gyr', 'Bubble_ID', 
        'Bubble_Center_X', 'Bubble_Center_Y', 
        'Bubble_Radius', 'Universe_Fraction_Remaining'
    ])

if __name__ == "__main__":
    # Generate the data
    df = generate_vacuum_decay_data()
    
    # Ensure output directory exists
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/raw/vacuum_decay_bubbles.csv'
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"Vacuum decay bubble data saved to {output_file}")
    print(f"Total rows: {len(df)}")
    print("Sample data:")
    print(df.head())
