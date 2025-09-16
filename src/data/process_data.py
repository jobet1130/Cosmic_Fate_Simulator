"""Data processing pipeline for Cosmic Fate Simulator.

This module provides functions to clean and preprocess astrophysical and cosmological data.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Unit conversion factors
KM_TO_M = 1000.0  # km to m
GYR_TO_YR = 1e9  # Gyr to years
MPC_TO_M = 3.086e22  # Mpc to meters

# Default configuration for different data types
DATA_CONFIG = {
    "cosmological_parameters": {
        "required_columns": ["Hubble_Constant_H0", "Omega_m", "Omega_Lambda", "w", "Omega_k"],
        "units": {
            "Hubble_Constant_H0": "km/s/Mpc",
            "Omega_m": "dimensionless",
            "Omega_Lambda": "dimensionless",
            "w": "dimensionless",
            "Omega_k": "dimensionless",
        },
        "scaling": "standard",
    },
    "galaxy_distances": {
        "required_columns": ["Galaxy_ID", "Initial_Distance_Mpc", "Time_Gyr", "Distance_Mpc"],
        "units": {
            "Initial_Distance_Mpc": "Mpc",
            "Time_Gyr": "Gyr",
            "Distance_Mpc": "Mpc",
        },
        "scaling": "minmax",
    },
    "vacuum_decay": {
        "required_columns": ["Simulation_ID", "Time_Gyr", "Bubble_ID", "Bubble_Radius", "Universe_Fraction_Remaining"],
        "units": {
            "Time_Gyr": "Gyr",
            "Bubble_Radius": "Mpc",
            "Universe_Fraction_Remaining": "dimensionless",
        },
        "scaling": "minmax",
    },
    "monte_carlo_times": {
        "required_columns": ["Simulation_ID", "Scenario", "H0", "Omega_m", "Omega_Lambda", "w", "Time_to_End_Gyr"],
        "units": {
            "H0": "km/s/Mpc",
            "Omega_m": "dimensionless",
            "Omega_Lambda": "dimensionless",
            "w": "dimensionless",
            "Time_to_End_Gyr": "Gyr",
        },
        "scaling": "standard",
    },
    "scale_factor_evolution": {
        "required_columns": ["Time_Gyr", "Scale_Factor", "Hubble_Parameter_H", "Deceleration_Parameter_q"],
        "units": {
            "Time_Gyr": "Gyr",
            "Scale_Factor": "dimensionless",
            "Hubble_Parameter_H": "km/s/Mpc",
            "Deceleration_Parameter_q": "dimensionless",
        },
        "scaling": "minmax",
    },
}


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {PROCESSED_DATA_DIR}")


def load_data(filename: str) -> pd.DataFrame:
    """Load data from raw data directory.
    
    Args:
        filename: Name of the file to load (with extension)
        
    Returns:
        Loaded DataFrame
    """
    filepath = RAW_DATA_DIR / filename
    try:
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        elif filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif filepath.suffix == ".h5":
            df = pd.read_hdf(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        threshold: Maximum fraction of missing values allowed in a column
        
    Returns:
        DataFrame with handled missing values
    """
    # Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    
    if cols_to_drop:
        logger.warning(f"Dropping columns with >{threshold*100}% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # For remaining columns, impute missing values with median (for numerical) or mode (for categorical)
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                method = "median"
            else:
                fill_value = df[col].mode()[0]
                method = "mode"
                
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with {method}")
    
    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.
    
    Args:
        df: Input DataFrame
        subset: List of columns to consider for identifying duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    n_duplicates = df.duplicated(subset=subset).sum()
    if n_duplicates > 0:
        df = df.drop_duplicates(subset=subset, keep="first")
        logger.info(f"Removed {n_duplicates} duplicate rows")
    return df


def convert_units(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Convert units to a consistent system.
    
    Args:
        df: Input DataFrame
        data_type: Type of data being processed (key in DATA_CONFIG)
        
    Returns:
        DataFrame with converted units
    """
    if data_type not in DATA_CONFIG:
        logger.warning(f"No unit conversion config found for {data_type}")
        return df
    
    unit_config = DATA_CONFIG[data_type].get("units", {})
    
    for col, unit in unit_config.items():
        if col not in df.columns:
            continue
            
        if unit == "km/s/Mpc":
            # Convert H0 from km/s/Mpc to 1/s
            df[col] = df[col] * KM_TO_M / MPC_TO_M
            logger.info(f"Converted {col} from km/s/Mpc to 1/s")
        elif unit == "Gyr":
            # Convert Gyr to years
            df[col] = df[col] * GYR_TO_YR
            logger.info(f"Converted {col} from Gyr to years")
    
    return df


def detect_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Detect and handle outliers using z-score method.
    
    Args:
        df: Input DataFrame
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    df_out = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = z_scores > z_threshold
        
        if outliers.any():
            # Cap outliers at the threshold
            median = df[col].median()
            mad = stats.median_abs_deviation(df[col], scale="normal")
            lower_bound = median - z_threshold * mad
            upper_bound = median + z_threshold * mad
            
            df_out[col] = df[col].clip(lower_bound, upper_bound)
            logger.info(f"Capped {outliers.sum()} outliers in column {col}")
    
    return df_out


def scale_features(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Scale features using the specified method.
    
    Args:
        df: Input DataFrame
        data_type: Type of data being processed (key in DATA_CONFIG)
        
    Returns:
        DataFrame with scaled features
    """
    if data_type not in DATA_CONFIG:
        return df
    
    scaling_method = DATA_CONFIG[data_type].get("scaling")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return df
    
    if scaling_method == "standard":
        # Standardization (mean=0, std=1)
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        logger.info(f"Standardized features: {numeric_cols}")
    elif scaling_method == "minmax":
        # Min-Max scaling to [0, 1]
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (
            df[numeric_cols].max() - df[numeric_cols].min()
        )
        logger.info(f"Min-Max scaled features: {numeric_cols}")
    
    return df


def validate_data(df: pd.DataFrame, data_type: str) -> bool:
    """Validate the processed data against expected schema.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data being processed (key in DATA_CONFIG)
        
    Returns:
        bool: True if validation passes, raises ValueError otherwise
    """
    if data_type not in DATA_CONFIG:
        logger.warning(f"No validation config found for {data_type}")
        return True
    
    required_cols = set(DATA_CONFIG[data_type].get("required_columns", []))
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns for {data_type}: {missing_cols}")
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        raise ValueError("Infinite values found in numeric columns")
    
    # Check for NaN values
    if df.isnull().any().any():
        raise ValueError("NaN values found in the data")
    
    logger.info(f"Validation passed for {data_type}")
    return True


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to the processed data directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename (with .parquet extension)
    """
    output_path = PROCESSED_DATA_DIR / filename
    
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    logger.info(f"Saved processed data to {output_path}")


def process_data_file(input_file: str, data_type: str, output_file: Optional[str] = None) -> None:
    """Process a single data file through the entire pipeline.
    
    Args:
        input_file: Input filename (must be in data/raw/)
        data_type: Type of data (key in DATA_CONFIG)
        output_file: Output filename (optional, defaults to input_file in processed/)
    """
    logger.info(f"Starting processing of {input_file} as {data_type}")
    
    if output_file is None:
        output_file = input_file
    
    try:
        # 1. Load data
        df = load_data(input_file)
        
        # 2. Handle missing values
        df = handle_missing_values(df)
        
        # 3. Remove duplicates
        df = remove_duplicates(df)
        
        # 4. Convert units
        df = convert_units(df, data_type)
        
        # 5. Handle outliers
        df = detect_outliers(df)
        
        # 6. Scale features
        df = scale_features(df, data_type)
        
        # 7. Validate data
        validate_data(df, data_type)
        
        # 8. Save processed data
        save_processed_data(df, output_file)
        
        logger.info(f"Successfully processed {input_file}")
        return df
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        raise


def main() -> None:
    """Main function to process all data files."""
    logger.info("Starting data processing pipeline")
    setup_directories()
    
    # Map of filename patterns to data types
    file_patterns = {
        'cosmological_parameters': 'cosmological_parameters',
        'galaxy_distance': 'galaxy_distances',
        'vacuum_decay': 'vacuum_decay',
        'monte_carlo': 'monte_carlo_times',
        'scale_factor': 'scale_factor_evolution'
    }
    
    # Process each file in the raw data directory
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.startswith('.') or not filename.endswith('.csv'):
            continue
            
        # Determine data type based on filename
        data_type = 'unknown'
        for pattern, dtype in file_patterns.items():
            if pattern.lower() in filename.lower():
                data_type = dtype
                break
                
        if data_type == 'unknown':
            logger.warning(f"Could not determine data type for {filename}, skipping")
            continue
            
        try:
            logger.info(f"Processing {filename} as {data_type}")
            process_data_file(filename, data_type)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}", exc_info=True)
            continue
    
    logger.info("Data processing pipeline completed")


if __name__ == "__main__":
    main()
