"""
Self-Healing ML System - Training Module
Trains models with configurable failure modes for testing the healing pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FAILURE INJECTION ENGINE
# ============================================================================

class FailureInjector:
    """Injects systematic failures into training data."""
    
    @staticmethod
    def inject_label_noise(X, y, noise_rate=0.2, condition_fn=None):
        """
        Flip labels in specific regions.
        
        Args:
            condition_fn: Function that returns True for samples to corrupt
                         e.g., lambda x: x[0] > 0.5 and x[2] < 0.3
        """
        y_corrupted = y.copy()
        n_classes = len(np.unique(y))
        
        if condition_fn is None:
            # Random noise
            mask = np.random.rand(len(y)) < noise_rate
        else:
            # Conditional noise (systematic)
            mask = np.array([condition_fn(x) for x in X])
            # Apply noise_rate within the condition
            mask = mask & (np.random.rand(len(y)) < noise_rate)
        
        # Flip labels
        for i in np.where(mask)[0]:
            available_labels = [c for c in range(n_classes) if c != y[i]]
            y_corrupted[i] = np.random.choice(available_labels)
        
        print(f"Injected label noise: {mask.sum()} samples ({mask.sum()/len(y)*100:.1f}%)")
        return X, y_corrupted, mask
    
    @staticmethod
    def inject_missing_feature(X, y, feature_idx=0):
        """Remove informative feature."""
        X_corrupted = X.copy()
        X_corrupted[:, feature_idx] = 0  # Zero out feature
        print(f"Removed feature {feature_idx}")
        return X_corrupted, y
    
    @staticmethod
    def inject_spurious_correlation(X, y, correlation_strength=0.9):
        """Add spurious feature correlated with label but not causal."""
        spurious_feature = np.zeros(len(y))
        
        # Create correlation with label
        for i in range(len(y)):
            if np.random.rand() < correlation_strength:
                spurious_feature[i] = y[i] + np.random.normal(0, 0.1)
            else:
                spurious_feature[i] = np.random.normal(0, 1)
        
        X_corrupted = np.column_stack([X, spurious_feature])
        print(f"Added spurious feature (correlation: {correlation_strength})")
        return X_corrupted, y
    
    @staticmethod
    def inject_data_shift(X_train, y_train, X_test, y_test, shift_type='covariate'):
        """
        Create distribution shift between train and test.
        
        Types:
        - 'covariate': P(X) changes
        - 'label': P(Y) changes  
        - 'concept': P(Y|X) changes
        """
        if shift_type == 'covariate':
            # Subsample training data from one region
            condition = X_train[:, 0] > np.median(X_train[:, 0])
            mask = condition
            X_train_shifted = X_train[mask]
            y_train_shifted = y_train[mask]
            
            # Test data includes all regions
            print(f"Covariate shift: Train size {len(X_train)} → {len(X_train_shifted)}")
            return X_train_shifted, y_train_shifted, X_test, y_test
        
        elif shift_type == 'label':
            # Imbalanced training, balanced test
            class_0_mask = y_train == 0
            class_1_mask = y_train == 1
            
            # Keep 80% of class 0, 20% of class 1
            keep_0 = np.random.rand(class_0_mask.sum()) < 0.8
            keep_1 = np.random.rand(class_1_mask.sum()) < 0.2
            
            mask = np.zeros(len(y_train), dtype=bool)
            mask[np.where(class_0_mask)[0][keep_0]] = True
            mask[np.where(class_1_mask)[0][keep_1]] = True
            
            X_train_shifted = X_train[mask]
            y_train_shifted = y_train[mask]
            
            print(f"Label shift: Class distribution changed")
            return X_train_shifted, y_train_shifted, X_test, y_test
        
        return X_train, y_train, X_test, y_test


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for tabular data."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=2, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SimpleCNN(nn.Module):
    """CNN for image data (MNIST)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_adult_income():
    """Load Adult Income dataset (binary classification)."""
    print("Loading Adult Income dataset...")
    
    # Create synthetic version for faster loading
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.05  # 5% label noise naturally
    )
    
    # Make features more interpretable
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }


def load_mnist():
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")
    
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset
        }
    except:
        print("MNIST download failed, using synthetic data")
        return None


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class Trainer:
    """Trains models with optional failure injection."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(loader), 100. * correct / total
    
    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(loader), 100. * correct / total
    
    def fit(self, train_loader, val_loader, epochs=20, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\nTraining on {self.device}")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        print("=" * 60)
        return self.history


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_model_with_failures(
    dataset_name='adult',
    failure_mode=None,
    failure_params=None,
    model_params=None,
    save_path='./checkpoints'
):
    """
    Main training function with failure injection.
    
    Args:
        dataset_name: 'adult' or 'mnist'
        failure_mode: None, 'label_noise', 'missing_feature', 'spurious', 'data_shift'
        failure_params: Dict of parameters for failure injection
        model_params: Dict of model hyperparameters
        save_path: Where to save checkpoints
    """
    
    Path(save_path).mkdir(exist_ok=True)
    
    # Default parameters
    if failure_params is None:
        failure_params = {}
    if model_params is None:
        model_params = {}
    
    # Load data
    if dataset_name == 'adult':
        data = load_adult_income()
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Inject failures
        failure_metadata = {}
        if failure_mode == 'label_noise':
            condition = failure_params.get('condition', lambda x: x[0] > 0.5)
            noise_rate = failure_params.get('noise_rate', 0.3)
            X_train, y_train, noise_mask = FailureInjector.inject_label_noise(
                X_train, y_train, noise_rate, condition
            )
            failure_metadata['noise_mask'] = noise_mask
        
        elif failure_mode == 'missing_feature':
            feature_idx = failure_params.get('feature_idx', 0)
            X_train, y_train = FailureInjector.inject_missing_feature(
                X_train, y_train, feature_idx
            )
            X_test, y_test = FailureInjector.inject_missing_feature(
                X_test, y_test, feature_idx
            )
            failure_metadata['removed_feature'] = feature_idx
        
        elif failure_mode == 'spurious':
            strength = failure_params.get('correlation_strength', 0.9)
            X_train, y_train = FailureInjector.inject_spurious_correlation(
                X_train, y_train, strength
            )
            X_test, y_test = FailureInjector.inject_spurious_correlation(
                X_test, y_test, strength
            )
            failure_metadata['spurious_feature_idx'] = X_train.shape[1] - 1
        
        elif failure_mode == 'data_shift':
            shift_type = failure_params.get('shift_type', 'covariate')
            X_train, y_train, X_test, y_test = FailureInjector.inject_data_shift(
                X_train, y_train, X_test, y_test, shift_type
            )
            failure_metadata['shift_type'] = shift_type
        
        # Create tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # Create model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = MLP(
            input_dim=input_dim,
            hidden_dims=model_params.get('hidden_dims', [64, 32]),
            num_classes=num_classes,
            dropout=model_params.get('dropout', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Train
    trainer = Trainer(model)
    history = trainer.fit(
        train_loader, 
        test_loader, 
        epochs=model_params.get('epochs', 20),
        lr=model_params.get('lr', 0.001)
    )
    
    # Save checkpoint
    
    # Remove non-serializable items (lambdas, functions)
    failure_params_serializable = {}
    if failure_params:
        for k, v in failure_params.items():
            if not callable(v):  # Skip functions/lambdas
                failure_params_serializable[k] = v
            else:
                failure_params_serializable[k] = str(v)  # Save as string representation
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'dataset_name': dataset_name,
        'failure_mode': failure_mode,
        'failure_params': failure_params_serializable,  # ← Fixed
        'failure_metadata': failure_metadata,
        'history': history,
        'final_val_acc': history['val_acc'][-1],
        'input_dim': input_dim if dataset_name == 'adult' else None,
        'num_classes': num_classes,
        'feature_names': data.get('feature_names', None)
    }
    
    checkpoint_path = Path(save_path) / f"model_{failure_mode or 'clean'}.pt"
    torch.save(checkpoint, checkpoint_path)
    # Save data for later analysis
    data_path = Path(save_path) / f"data_{failure_mode or 'clean'}.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'failure_metadata': failure_metadata
        }, f)
    
    print(f"\n✓ Model saved to {checkpoint_path}")
    print(f"✓ Data saved to {data_path}")
    print(f"✓ Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    return checkpoint, model, trainer


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("Self-Healing ML System - Training Module")
    print("=" * 60)
    
    # Train clean baseline
    print("\n[1/5] Training CLEAN baseline model...")
    train_model_with_failures(
        dataset_name='adult',
        failure_mode=None,
        save_path='./checkpoints'
    )
    
    # Train with label noise
    print("\n[2/5] Training model with LABEL NOISE...")
    train_model_with_failures(
        dataset_name='adult',
        failure_mode='label_noise',
        failure_params={
            'condition': lambda x: x[0] > 0.5 and x[1] < 0,  # Conditional noise
            'noise_rate': 0.3
        },
        save_path='./checkpoints'
    )
    
    # Train with missing feature
    print("\n[3/5] Training model with MISSING FEATURE...")
    train_model_with_failures(
        dataset_name='adult',
        failure_mode='missing_feature',
        failure_params={'feature_idx': 0},
        save_path='./checkpoints'
    )
    
    # Train with spurious correlation
    print("\n[4/5] Training model with SPURIOUS CORRELATION...")
    train_model_with_failures(
        dataset_name='adult',
        failure_mode='spurious',
        failure_params={'correlation_strength': 0.9},
        save_path='./checkpoints'
    )
    
    # Train with data shift
    print("\n[5/5] Training model with DATA SHIFT...")
    train_model_with_failures(
        dataset_name='adult',
        failure_mode='data_shift',
        failure_params={'shift_type': 'covariate'},
        save_path='./checkpoints'
    )
    
    print("\n" + "=" * 60)
    print("✓ All models trained successfully!")
    print("✓ Checkpoints saved to ./checkpoints/")
    print("\nNext: Run failure discovery module on these models.")