"""
Model Training Script for AI Fake News Detector
Fine-tunes transformer models (BERT/RoBERTa) for binary classification
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import json
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FakeNewsTrainer:
    """Handles model training and evaluation"""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        self.model.to(self.device)
        
        # TensorBoard writer
        self.writer = SummaryWriter('runs/fake_news_detector')
        
    def prepare_data(
        self, 
        data_path: str, 
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare data for training
        
        Args:
            data_path: Path to processed CSV file
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Create dataset
        dataset = FakeNewsDataset(
            texts=df['text'].values,
            labels=df['label'].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Split into train and validation
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """
        Evaluate model on validation set
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate
            metrics = self.evaluate(val_loader)
            
            logger.info(f"Val Loss: {metrics['loss']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/val', metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/val', metrics['f1'], epoch)
            
            # Save best model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.save_model('ml/models/best_model.pt', metrics)
                logger.info(f"Saved new best model with F1: {best_f1:.4f}")
        
        self.writer.close()
    
    def save_model(self, path: str, metrics: Dict = None):
        """
        Save model and tokenizer
        
        Args:
            path: Path to save model
            metrics: Optional metrics to save
        """
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'metrics': metrics
        }, path)
        
        # Save tokenizer
        tokenizer_dir = save_dir / 'tokenizer'
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'metrics': metrics
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    # Training configuration
    TRAIN_CONFIG = {
        'model_name': 'roberta-base',  # or 'bert-base-uncased'
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3
    }
    
    # Initialize trainer
    trainer = FakeNewsTrainer(**TRAIN_CONFIG)
    
    # Prepare data
    data_path = 'ml/data/processed/processed_data.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run data_loader.py first to create the dataset")
    else:
        train_loader, val_loader = trainer.prepare_data(data_path)
        
        # Train model
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader)
        
        logger.info("Training complete!")
