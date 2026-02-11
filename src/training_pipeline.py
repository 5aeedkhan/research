import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json
from tqdm import tqdm

from .fusion_model import MultiFeatureFusionModel
from .shap_xgboost import SHAPAwareXGBoost
from .audio_preprocessor import AudioPreprocessor, SpeechDataset

class SpeechDisorderTrainer:
    def __init__(self, 
                 model: MultiFeatureFusionModel,
                 device: torch.device,
                 save_dir: str = "checkpoints"):
        """
        Trainer for speech disorder classification model.
        
        Args:
            model: The neural network model
            device: Device to train on (cpu/cuda)
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            mel_spec = batch['mel_spectrogram'].to(self.device)
            mfcc = batch['mfcc'].to(self.device)
            features = {k: v.to(self.device) for k, v in batch['features'].items()}
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(mel_spec, mfcc, features)
            logits = outputs['logits']
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, dataloader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                mel_spec = batch['mel_spectrogram'].to(self.device)
                mfcc = batch['mfcc'].to(self.device)
                features = {k: v.to(self.device) for k, v in batch['features'].items()}
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(mel_spec, mfcc, features)
                logits = outputs['logits']
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100. * correct / total
        
        # Detailed metrics
        report = classification_report(
            all_labels, all_predictions, 
            target_names=['healthy', 'dysarthria', 'apraxia', 'dysphonia'],
            output_dict=True
        )
        
        return epoch_loss, epoch_accuracy, report
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, learning_rate: float = 1e-4,
              patience: int = 10, weight_decay: float = 1e-5) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            weight_decay: Weight decay for regularization
        """
        # Setup optimizer and criterion
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc, val_report = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(self.save_dir, 'best_model.pth'))
                
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training history
        self.save_training_history()
        
        return {
            'best_val_accuracy': best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_training_history(self):
        """Save training history to file."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.show()

class CompletePipeline:
    """Complete pipeline for speech disorder classification."""
    
    def __init__(self, device: torch.device, save_dir: str = "results"):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.model = None
        self.trainer = None
        self.shap_xgb = None
        
    def setup_model(self, num_classes: int = 4):
        """Setup the neural network model."""
        self.model = MultiFeatureFusionModel(
            input_dim=128,
            seq_len=256,
            cnn_feature_dim=512,
            transformer_dim=512,
            num_classes=num_classes
        )
        
        self.trainer = SpeechDisorderTrainer(self.model, self.device, self.save_dir)
        
    def train_neural_model(self, dataset: SpeechDataset, 
                          test_size: float = 0.2, 
                          batch_size: int = 16,
                          num_epochs: int = 50) -> Dict:
        """Train the neural network model."""
        # Split dataset
        dataset_size = len(dataset)
        val_size = int(test_size * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        history = self.trainer.train(
            train_loader, val_loader, 
            num_epochs=num_epochs,
            learning_rate=1e-4,
            patience=15
        )
        
        # Plot training history
        self.trainer.plot_training_history()
        
        return history
    
    def train_xgboost_classifier(self, dataset: SpeechDataset,
                                batch_size: int = 32) -> Dict:
        """Train SHAP-aware XGBoost classifier using neural model features."""
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features using trained neural model
        X, y = self.shap_xgb.extract_features_from_neural_model(
            self.model, dataloader, self.device
        )
        
        # Generate feature names
        feature_names = []
        for i in range(512):  # fused features
            feature_names.append(f'fused_{i}')
        for i in range(512):  # CNN features
            feature_names.append(f'cnn_{i}')
        for i in range(512):  # Transformer features
            feature_names.append(f'transformer_{i}')
        for i in range(512):  # Acoustic features
            feature_names.append(f'acoustic_{i}')
        
        class_names = ['healthy', 'dysarthria', 'apraxia', 'dysphonia']
        
        # Train XGBoost
        self.shap_xgb.fit(X, y, feature_names=feature_names, class_names=class_names)
        
        # Evaluate
        results = self.shap_xgb.evaluate(X, y)
        
        return results
    
    def complete_training(self, dataset: SpeechDataset) -> Dict:
        """Run complete training pipeline."""
        print("=== Starting Complete Training Pipeline ===")
        
        # Setup model
        print("Setting up model...")
        self.setup_model()
        
        # Train neural model
        print("\n=== Training Neural Network Model ===")
        neural_history = self.train_neural_model(dataset)
        
        # Setup XGBoost
        print("\n=== Training SHAP-aware XGBoost ===")
        self.shap_xgb = SHAPAwareXGBoost(use_gpu=torch.cuda.is_available())
        xgb_results = self.train_xgboost_classifier(dataset)
        
        # Combine results
        complete_results = {
            'neural_network': neural_history,
            'xgboost': xgb_results,
            'final_accuracy': xgb_results['accuracy'],
            'cv_score': xgb_results['cv_mean']
        }
        
        # Save complete results
        with open(os.path.join(self.save_dir, 'complete_results.json'), 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\n=== Training Complete ===")
        print(f"Final XGBoost Accuracy: {xgb_results['accuracy']:.4f}")
        print(f"Cross-validation Score: {xgb_results['cv_mean']:.4f} ± {xgb_results['cv_std']:.4f}")
        
        return complete_results

# Test the training pipeline
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # This would be used with actual data
    print("Training pipeline implemented successfully!")
    print("Use with actual speech disorder dataset to train the complete model.")
