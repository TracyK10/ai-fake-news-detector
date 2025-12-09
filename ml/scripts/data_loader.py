"""
Data Loader Module for AI Fake News Detector
Handles loading and merging datasets from multiple sources
"""

import pandas as pd
import os
from typing import Tuple, Optional
from loguru import logger
import requests
from pathlib import Path


class DataLoader:
    """Loads and merges fake news datasets from various sources"""
    
    def __init__(self, data_dir: str = "ml/data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, file_path: str, text_col: str, label_col: str) -> pd.DataFrame:
        """
        Load a CSV dataset and standardize column names
        
        Args:
            file_path: Path to CSV file
            text_col: Name of the text column
            label_col: Name of the label column
            
        Returns:
            DataFrame with standardized 'text' and 'label' columns
        """
        try:
            df = pd.read_csv(file_path)
            df = df[[text_col, label_col]].copy()
            df.columns = ['text', 'label']
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def normalize_labels(self, df: pd.DataFrame, 
                        fake_values: list, 
                        real_values: list) -> pd.DataFrame:
        """
        Normalize labels to binary format (0=Real, 1=Fake)
        
        Args:
            df: Input DataFrame
            fake_values: List of values that represent fake news
            real_values: List of values that represent real news
            
        Returns:
            DataFrame with normalized labels
        """
        df_copy = df.copy()
        
        # Map fake values to 1
        df_copy.loc[df_copy['label'].isin(fake_values), 'label'] = 1
        
        # Map real values to 0
        df_copy.loc[df_copy['label'].isin(real_values), 'label'] = 0
        
        # Remove any rows with labels that aren't 0 or 1
        df_copy = df_copy[df_copy['label'].isin([0, 1])].copy()
        
        # Convert to int
        df_copy['label'] = df_copy['label'].astype(int)
        
        logger.info(f"Normalized {len(df_copy)} records")
        return df_copy
    
    def merge_datasets(self, datasets: list) -> pd.DataFrame:
        """
        Merge multiple datasets into one
        
        Args:
            datasets: List of DataFrames to merge
            
        Returns:
            Merged DataFrame
        """
        if not datasets:
            logger.warning("No datasets to merge")
            return pd.DataFrame()
        
        merged = pd.concat(datasets, ignore_index=True)
        
        # Remove duplicates based on text
        merged = merged.drop_duplicates(subset=['text'], keep='first')
        
        # Remove null values
        merged = merged.dropna()
        
        # Reset index
        merged = merged.reset_index(drop=True)
        
        logger.info(f"Merged dataset contains {len(merged)} records")
        logger.info(f"Label distribution:\n{merged['label'].value_counts()}")
        
        return merged
    
    def load_kaggle_fake_news(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load Kaggle Fake News dataset
        Expected columns: 'text' and 'label' where label is 0 (real) or 1 (fake)
        
        Args:
            file_path: Optional custom path to the dataset
            
        Returns:
            Standardized DataFrame
        """
        if file_path is None:
            file_path = self.data_dir / "kaggle_fake_news.csv"
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            logger.info("Please download from: https://www.kaggle.com/c/fake-news/data")
            return pd.DataFrame()
        
        df = self.load_csv(file_path, text_col='text', label_col='label')
        df = self.normalize_labels(df, fake_values=[1, '1'], real_values=[0, '0'])
        
        return df
    
    def load_liar_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load LIAR dataset
        Expected format: TSV with statement and label columns
        Labels: pants-fire, false, barely-true -> Fake (1)
                half-true, mostly-true, true -> Real (0)
        
        Args:
            file_path: Optional custom path to the dataset
            
        Returns:
            Standardized DataFrame
        """
        if file_path is None:
            file_path = self.data_dir / "liar_dataset.tsv"
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            logger.info("Please download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
            return pd.DataFrame()
        
        try:
            # LIAR dataset has no header, columns are:
            # 0: ID, 1: label, 2: statement, ...
            df = pd.read_csv(file_path, sep='\t', header=None)
            df = df[[2, 1]].copy()  # statement, label
            df.columns = ['text', 'label']
            
            # Normalize labels
            fake_labels = ['pants-fire', 'false', 'barely-true']
            real_labels = ['half-true', 'mostly-true', 'true']
            
            df = self.normalize_labels(df, fake_values=fake_labels, real_values=real_labels)
            
            logger.info(f"Loaded LIAR dataset: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading LIAR dataset: {e}")
            return pd.DataFrame()
    
    def create_sample_dataset(self) -> pd.DataFrame:
        """
        Create a small sample dataset for testing purposes
        
        Returns:
            Sample DataFrame
        """
        samples = [
            {"text": "Scientists discover new cure for cancer in groundbreaking study", "label": 0},
            {"text": "President announces new economic policy to boost job growth", "label": 0},
            {"text": "SHOCKING: Aliens land in New York City and nobody noticed!", "label": 1},
            {"text": "You won't believe what this celebrity did next - doctors hate them!", "label": 1},
            {"text": "Climate change report shows continued warming trends globally", "label": 0},
            {"text": "BREAKING: Secret government facility discovered on the moon", "label": 1},
        ]
        
        df = pd.DataFrame(samples)
        logger.info(f"Created sample dataset with {len(df)} records")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """
        Save processed dataset to disk
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_dir = Path("ml/data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} records to {output_path}")

    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load and merge all available datasets
        
        Returns:
            Merged DataFrame
        """
        datasets = []
        
        # Try to load each dataset
        kaggle_df = self.load_kaggle_fake_news()
        if not kaggle_df.empty:
            datasets.append(kaggle_df)
        
        liar_df = self.load_liar_dataset()
        if not liar_df.empty:
            datasets.append(liar_df)
        
        # If no datasets found, create sample
        if not datasets:
            logger.warning("No datasets found, creating sample dataset")
            datasets.append(self.create_sample_dataset())
        
        return self.merge_datasets(datasets)


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load all datasets
    df = loader.load_all_datasets()
    
    # Save processed data
    if not df.empty:
        loader.save_processed_data(df)
        print(f"\nDataset Summary:")
        print(f"Total records: {len(df)}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"\nSample records:")
        print(df.head())
