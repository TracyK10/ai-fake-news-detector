"""
Model Inference Module for AI Fake News Detector
Handles loading and running inference with trained models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json
from loguru import logger
from typing import Dict, Tuple
import numpy as np


class FakeNewsClassifier:
    """Handles inference for fake news detection"""
    
    def __init__(self, model_path: str = "ml/models/best_model.pt"):
        """
        Initialize classifier with trained model
        
        Args:
            model_path: Path to saved model checkpoint
        """
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load config
        config_path = self.model_path.parent / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        tokenizer_dir = self.model_path.parent / 'tokenizer'
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        if 'metrics' in checkpoint and checkpoint['metrics']:
            logger.info(f"Model metrics: {checkpoint['metrics']}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict if text is fake or real news
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.get('max_length', 512),
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get predictions
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Map to labels
        label_map = {0: "Real", 1: "Fake"}
        label = label_map[predicted_class]
        
        result = {
            'label': label,
            'confidence_score': float(confidence),
            'probabilities': {
                'real': float(probabilities[0][0]),
                'fake': float(probabilities[0][1])
            }
        }
        
        return result
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict multiple texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the classifier
    classifier = FakeNewsClassifier()
    
    # Test samples
    test_texts = [
        "Scientists at MIT have discovered a new treatment for cancer that shows promising results in clinical trials.",
        "SHOCKING: Aliens have landed in New York and the government is hiding it from us!",
        "The stock market closed higher today as investors responded positively to economic data.",
    ]
    
    print("Fake News Detection Results:\n")
    for i, text in enumerate(test_texts, 1):
        result = classifier.predict(text)
        print(f"Text {i}: {text[:80]}...")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Probabilities: Real={result['probabilities']['real']:.2%}, Fake={result['probabilities']['fake']:.2%}\n")
