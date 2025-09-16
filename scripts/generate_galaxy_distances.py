import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def big_freeze_scale_factor(t, H0=70):
    """Exponential growth for Big Freeze scenario"""
    return np.exp(H0 * t / 100)

def big_rip_scale_factor(t, H0=70):
    """Rapidly diverging scale factor for Big Rip scenario"""
    t_rip = 150  # Time of the Big Rip
    return 1 / (1 - (t/t_rip)**2) if t < t_rip else np.inf

def big_crunch_scale_factor(t, t_crunch=150):
    """Scale factor that grows then collapses to 0 for Big Crunch"""
    return np.sin(np.pi * t / (2 * t_crunch)) if t < t_crunch else 0

def generate_galaxy_distances():
    # Parameters
    n_galaxies = 1000
    time_steps = np.arange(0, 101, 1)  # 0 to 100 Gyr in steps of 1 Gyr
    scenarios = ['Big Freeze', 'Big Rip', 'Big Crunch']
    
    # Initialize list to store all data
    all_data = []
    
    # Generate data for each galaxy
    for galaxy_id in range(1, n_galaxies + 1):
        # Random initial distance between 1 and 5000 Mpc
        initial_distance = np.random.uniform(1, 5000)
        
        # Randomly assign a scenario to this galaxy
        scenario = np.random.choice(scenarios)
        
        # Generate distance evolution for this galaxy
        for t in time_steps:
            # Get scale factor at time t
            if scenario == 'Big Freeze':
                a_t = big_freeze_scale_factor(t)
            elif scenario == 'Big Rip':
                a_t = big_rip_scale_factor(t)
                if np.isinf(a_t):  # After Big Rip, distance becomes infinite
                    distance = np.inf
                    all_data.append([galaxy_id, initial_distance, t, distance, scenario])
                    break  # No need to continue after Big Rip
            else:  # Big Crunch
                a_t = big_crunch_scale_factor(t)
                if a_t <= 0:  # After Big Crunch
                    distance = 0
                    all_data.append([galaxy_id, initial_distance, t, distance, scenario])
                    break  # No need to continue after Big Crunch
            
            # Calculate current distance (scale by the scale factor)
            distance = initial_distance * a_t
            
            # Add to data
            all_data.append([galaxy_id, initial_distance, t, distance, scenario])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data, columns=['Galaxy_ID', 'Initial_Distance_Mpc', 'Time_Gyr', 'Distance_Mpc', 'Scenario'])
    return df

if __name__ == "__main__":
    # Generate the data
    df = generate_galaxy_distances()
    
    # Ensure output directory exists
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/raw/galaxy_distance_evolution.csv'
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Galaxy distance evolution data saved to {output_file}")
    print(f"Total rows: {len(df)}")
    print("Sample data:")
    print(df.head())
