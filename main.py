import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
import torch.nn.functional as F

from utils.data_loader import HandwritingDataset
from models.model import get_model

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MixupAugmentation:
    """Mixup augmentation for better generalization"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class HandwritingForgeryTrainer:
    def __init__(self, dataset_path, model_type='custom', batch_size=16, learning_rate=0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Load dataset with enhanced augmentation
        self.dataset = HandwritingDataset(dataset_path, augment=True)
        
        # Split dataset with stratification
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        
        # Create data loaders with more workers
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize model
        self.model = get_model(model_type=model_type, num_classes=2).to(self.device)
        
        # Enhanced loss function and optimizer
        self.criterion = FocalLoss(alpha=1, gamma=2)  # Focal loss for better handling of class imbalance
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
        
        # Advanced learning rate scheduling
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Mixup augmentation
        self.mixup = MixupAugmentation(alpha=0.2)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch with enhanced techniques"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup augmentation
            data, targets_a, targets_b, lam = self.mixup(data, target)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Mixup loss
            loss = lam * self.criterion(output, targets_a) + (1 - lam) * self.criterion(output, targets_b)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy (use original targets for evaluation)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, epochs=100, early_stopping_patience=15):
        """Train the model with enhanced techniques"""
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Using Focal Loss with Mixup Augmentation")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, predictions, targets = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Enhanced early stopping based on both loss and accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, 'best_model.pth')
                print(f"New best model saved! Accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Load best model
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with accuracy {checkpoint['val_acc']:.2f}%")
        
        return predictions, targets
    
    def evaluate(self):
        """Evaluate the model and print detailed metrics"""
        val_loss, val_acc, predictions, targets = self.validate()
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
        conf_matrix = confusion_matrix(targets, predictions)
        
        print("\n" + "="*60)
        print("FINAL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Genuine  Forged")
        print(f"Actual Genuine    {conf_matrix[0][0]:6d}  {conf_matrix[0][1]:6d}")
        print(f"      Forged      {conf_matrix[1][0]:6d}  {conf_matrix[1][1]:6d}")
        
        # Calculate per-class accuracy
        genuine_acc = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]) * 100
        forged_acc = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1]) * 100
        print(f"\nPer-class Accuracy:")
        print(f"Genuine: {genuine_acc:.2f}%")
        print(f"Forged: {forged_acc:.2f}%")
        
        return {
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'genuine_accuracy': genuine_acc,
            'forged_accuracy': forged_acc
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.8)
        ax1.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', alpha=0.8)
        ax2.plot(self.val_accuracies, label='Validation Accuracy', alpha=0.8)
        ax2.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% Target')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration for 90%+ accuracy
    DATASET_PATH = "dataset"
    MODEL_TYPE = "custom"  # Use custom model for best performance
    BATCH_SIZE = 16  # Smaller batch size for better generalization
    LEARNING_RATE = 0.0001  # Lower learning rate for stable training
    EPOCHS = 100  # More epochs for convergence
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path {DATASET_PATH} not found!")
        return
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize trainer
    trainer = HandwritingForgeryTrainer(
        dataset_path=DATASET_PATH,
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Train the model
    predictions, targets = trainer.train(epochs=EPOCHS)
    
    # Evaluate
    results = trainer.evaluate()
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\nTraining completed successfully!")
    print(f"Best model saved as 'best_model.pth'")
    print(f"Training history plot saved as 'training_history.png'")
    
    # Check if 90% accuracy was achieved
    if results['accuracy'] >= 90:
        print(f"\n🎉 SUCCESS! Achieved {results['accuracy']:.2f}% accuracy (Target: 90%+)")
    else:
        print(f"\n⚠️  Achieved {results['accuracy']:.2f}% accuracy (Target: 90%+)")
        print("Consider:")
        print("- Training for more epochs")
        print("- Using the ResNet model (MODEL_TYPE='resnet')")
        print("- Collecting more training data")
        print("- Adjusting hyperparameters")

if __name__ == "__main__":
    main()
