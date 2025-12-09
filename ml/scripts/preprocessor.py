"""
Text Preprocessing Module for AI Fake News Detector
Handles text cleaning and tokenization for NLP models
"""

import re
import string
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from loguru import logger
import pandas as pd


class TextPreprocessor:
    """Handles text cleaning and preprocessing for fake news detection"""
    
    def __init__(self, remove_stopwords: bool = False, lowercase: bool = True):
        """
        Initialize preprocessor
        
        Args:
            remove_stopwords: Whether to remove English stopwords
            lowercase: Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        # Download NLTK data if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text
        
        Args:
            text: Input text
            
        Returns:
            Text without HTML tags
        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        # Remove http/https URLs
        text = re.sub(r'http\S+|https\S+', '', text)
        # Remove www URLs
        text = re.sub(r'www\.\S+', '', text)
        return text
    
    def remove_mentions_hashtags(self, text: str) -> str:
        """
        Remove Twitter-style mentions and hashtags
        
        Args:
            text: Input text
            
        Returns:
            Text without mentions and hashtags
        """
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove #hashtags
        text = re.sub(r'#\w+', '', text)
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters and digits, keep only letters and spaces
        
        Args:
            text: Input text
            
        Returns:
            Text with only letters and spaces
        """
        # Keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace from text
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        return ' '.join(text.split())
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove mentions and hashtags
        text = self.remove_mentions_hashtags(text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess all text in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with cleaned text
        """
        df_copy = df.copy()
        
        logger.info(f"Preprocessing {len(df_copy)} text samples...")
        
        # Apply cleaning to each text
        df_copy[text_column] = df_copy[text_column].apply(self.clean_text)
        
        # Remove empty texts
        original_len = len(df_copy)
        df_copy = df_copy[df_copy[text_column].str.len() > 0].copy()
        removed = original_len - len(df_copy)
        
        if removed > 0:
            logger.info(f"Removed {removed} empty text samples")
        
        logger.info(f"Preprocessing complete. {len(df_copy)} samples remaining.")
        
        return df_copy
    
    def get_statistics(self, df: pd.DataFrame, text_column: str = 'text') -> Dict[str, Any]:
        """
        Get statistics about the preprocessed text
        
        Args:
            df: DataFrame with text
            text_column: Name of the text column
            
        Returns:
            Dictionary with statistics
        """
        texts = df[text_column].values
        
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        stats = {
            'num_samples': len(df),
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'max_word_count': max(word_counts) if word_counts else 0,
            'min_word_count': min(word_counts) if word_counts else 0,
            'avg_char_count': sum(char_counts) / len(char_counts) if char_counts else 0,
            'max_char_count': max(char_counts) if char_counts else 0,
            'min_char_count': min(char_counts) if char_counts else 0,
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(remove_stopwords=False, lowercase=True)
    
    # Test samples
    samples = [
        "<html>BREAKING NEWS: Check out this link https://fake-news.com #fakenews</html>",
        "Scientists at @MIT discover new treatment for cancer! More info at www.example.com",
        "This is a normal sentence with some UPPERCASE words and numbers like 123.",
    ]
    
    print("Text Preprocessing Examples:\n")
    for i, sample in enumerate(samples, 1):
        cleaned = preprocessor.clean_text(sample)
        print(f"Original {i}: {sample}")
        print(f"Cleaned {i}:  {cleaned}\n")
    
    # Test on DataFrame
    df = pd.DataFrame({
        'text': samples,
        'label': [1, 0, 0]
    })
    
    df_cleaned = preprocessor.preprocess_dataframe(df)
    
    print("\nDataFrame after preprocessing:")
    print(df_cleaned)
    
    print("\nText Statistics:")
    stats = preprocessor.get_statistics(df_cleaned)
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
