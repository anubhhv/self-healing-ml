"""
Module 2: Root Cause Analysis Engine
Diagnoses WHY failures occur using causal inference, influence functions, and ablation tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import ks_2samp, entropy
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional
import pickle
from dataclasses import dataclass
from train import MLP, Trainer
from failure_discovery import FailureCluster


@dataclass
class RootCause:
    """Represents a diagnosed root cause."""
    cause_type: str  # 'data_shift', 'label_noise', 'missing_feature', etc.
    confidence: float  # 0-1
    evidence: Dict
    affected_samples: np.ndarray
    recommendation: str


class DataShiftDetector:
    """Detects if failures are due to distribution shift."""
    
    def detect(self, X_train, X_test, failures):
        """
        Test if P(X|failure) != P(X|train)
        
        Returns:
            shift_score: 0-1, higher means more shift
            evidence: Dict with test statistics
        """
        failed_samples = X_test[failures]
        
        if len(failed_samples) < 10:
            return 0.0, {'reason': 'too_few_failures'}
        
        # Test each feature for distribution shift
        shift_scores = []
        feature_shifts = {}
        
        for feat_idx in range(X_train.shape[1]):
            train_feat = X_train[:, feat_idx]
            failed_feat = failed_samples[:, feat_idx]
            
            # Kolmogorov-Smirnov test
            statistic, pvalue = ks_2samp(train_feat, failed_feat)
            
            shift_scores.append(statistic)
            feature_shifts[f'feature_{feat_idx}'] = {
                'ks_statistic': float(statistic),
                'p_value': float(pvalue),
                'train_mean': float(train_feat.mean()),
                'failed_mean': float(failed_feat.mean()),
                'shift_magnitude': float(abs(failed_feat.mean() - train_feat.mean()))
            }
        
        # Overall shift score
        avg_shift = np.mean(shift_scores)
        max_shift = np.max(shift_scores)
        
        # Compute KL divergence for overall distribution
        train_hist, bins = np.histogram(X_train.flatten(), bins=50, density=True)
        failed_hist, _ = np.histogram(failed_samples.flatten(), bins=bins, density=True)
        
        train_hist += 1e-10  # Avoid log(0)
        failed_hist += 1e-10
        
        kl_div = entropy(failed_hist, train_hist)
        
        evidence = {
            'avg_shift': float(avg_shift),
            'max_shift': float(max_shift),
            'kl_divergence': float(kl_div),
            'feature_shifts': feature_shifts,
            'num_significant_features': sum(1 for fs in feature_shifts.values() if fs['p_value'] < 0.05)
        }
        
        # Higher shift score if multiple features shifted significantly
        shift_score = min(1.0, avg_shift * 2 + (kl_div / 10))
        
        return shift_score, evidence


class LabelNoiseDetector:
    """Detects if failures are due to noisy training labels."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def detect(self, X_train, y_train, X_test, y_test, failures):
        """
        Use influence functions to find suspicious training labels.
        
        Simplified version: Check if influential samples have high training loss.
        """
        if failures.sum() < 5:
            return 0.0, {'reason': 'too_few_failures'}
        
        # Get failed samples
        X_failed = X_test[failures]
        y_failed = y_test[failures]
        
        # Compute training losses
        self.model.eval()
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_train_t)
            losses = F.cross_entropy(outputs, y_train_t, reduction='none')
            train_losses = losses.cpu().numpy()
        
        # Find high-loss training samples (potential label noise)
        high_loss_threshold = np.percentile(train_losses, 90)
        suspicious_samples = train_losses > high_loss_threshold
        
        # Check if failed test samples are near suspicious training samples
        # Simple heuristic: compute nearest neighbor distance
        from sklearn.metrics import pairwise_distances
        
        distances = pairwise_distances(X_failed, X_train)
        nearest_train_idx = distances.argmin(axis=1)
        
        # What fraction of failed samples have suspicious nearest neighbors?
        noise_score = suspicious_samples[nearest_train_idx].mean()
        
        evidence = {
            'num_suspicious_train_samples': int(suspicious_samples.sum()),
            'pct_suspicious': float(suspicious_samples.mean() * 100),
            'avg_train_loss': float(train_losses.mean()),
            'high_loss_threshold': float(high_loss_threshold),
            'noise_score': float(noise_score),
            'failed_near_suspicious': float(noise_score * 100)
        }
        
        return noise_score, evidence
    
    def detect_label_inconsistency(self, X_train, y_train):
        """
        Find training samples with inconsistent labels.
        Uses k-NN: if neighbors have different labels, might be noise.
        """
        from sklearn.neighbors import NearestNeighbors
        
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(X_train)
        
        distances, indices = knn.kneighbors(X_train)
        
        inconsistency_scores = []
        for i in range(len(X_train)):
            neighbor_labels = y_train[indices[i]]
            # Inconsistency = fraction of neighbors with different label
            inconsistency = (neighbor_labels != y_train[i]).mean()
            inconsistency_scores.append(inconsistency)
        
        inconsistency_scores = np.array(inconsistency_scores)
        
        return inconsistency_scores


class ModelCapacityTester:
    """Tests if failures are due to insufficient model capacity."""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def test(self, X_train, y_train, X_test, y_test, failures, original_model):
        """
        Train larger model on failure cluster.
        If it improves significantly → underfitting/capacity issue.
        """
        if failures.sum() < 20:
            return 0.0, {'reason': 'too_few_failures'}
        
        # Create larger model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        large_model = MLP(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],  # Larger than original [64, 32]
            num_classes=num_classes,
            dropout=0.1
        )
        
        # Train on full data
        from torch.utils.data import TensorDataset, DataLoader
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=128)
        
        trainer = Trainer(large_model, device=self.device)
        trainer.fit(train_loader, test_loader, epochs=10, lr=0.001)
        
        # Evaluate on failure cluster
        large_model.eval()
        X_failed = torch.FloatTensor(X_test[failures]).to(self.device)
        y_failed = torch.LongTensor(y_test[failures]).to(self.device)
        
        with torch.no_grad():
            # Original model performance
            original_model.eval()
            orig_output = original_model(X_failed)
            orig_pred = orig_output.argmax(dim=1)
            orig_acc = (orig_pred == y_failed).float().mean().item()
            
            # Large model performance
            large_output = large_model(X_failed)
            large_pred = large_output.argmax(dim=1)
            large_acc = (large_pred == y_failed).float().mean().item()
        
        improvement = large_acc - orig_acc
        
        # Also check training vs validation loss for overfitting
        train_losses = trainer.history['train_loss']
        val_losses = trainer.history['val_loss']
        
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        overfit_gap = final_val_loss - final_train_loss
        
        evidence = {
            'original_acc_on_failures': float(orig_acc * 100),
            'large_model_acc_on_failures': float(large_acc * 100),
            'improvement': float(improvement * 100),
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'overfit_gap': float(overfit_gap)
        }
        
        # Capacity issue if large model significantly improves
        capacity_score = max(0, min(1, improvement * 5))  # Scale to 0-1
        
        # Determine if underfitting or overfitting
        if capacity_score > 0.5:
            evidence['diagnosis'] = 'underfitting'
        elif overfit_gap > 0.2:
            evidence['diagnosis'] = 'overfitting'
        else:
            evidence['diagnosis'] = 'appropriate_capacity'
        
        return capacity_score, evidence


class SpuriousCorrelationDetector:
    """Detects if model relies on spurious correlations."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def detect(self, X_train, y_train, X_test, y_test, failures):
        """
        Use feature importance + ablation to find spurious features.
        """
        if failures.sum() < 10:
            return 0.0, {'reason': 'too_few_failures'}
        
        # Compute feature importance using gradient-based method
        feature_importance = self._compute_feature_importance(X_test[failures], y_test[failures])
        
        # Test each feature by ablation
        ablation_results = {}
        
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.LongTensor(y_test).to(self.device)
        
        # Baseline accuracy
        self.model.eval()
        with torch.no_grad():
            baseline_output = self.model(X_test_t)
            baseline_pred = baseline_output.argmax(dim=1)
            baseline_acc = (baseline_pred == y_test_t).float().mean().item()
        
        # Ablate each feature
        for feat_idx in range(X_test.shape[1]):
            X_ablated = X_test.copy()
            X_ablated[:, feat_idx] = X_train[:, feat_idx].mean()  # Replace with train mean
            
            X_ablated_t = torch.FloatTensor(X_ablated).to(self.device)
            
            with torch.no_grad():
                ablated_output = self.model(X_ablated_t)
                ablated_pred = ablated_output.argmax(dim=1)
                ablated_acc = (ablated_pred == y_test_t).float().mean().item()
            
            acc_change = ablated_acc - baseline_acc
            
            ablation_results[f'feature_{feat_idx}'] = {
                'importance': float(feature_importance[feat_idx]),
                'ablation_acc_change': float(acc_change * 100),
                'is_spurious_candidate': acc_change > 0.05  # Accuracy improves when removed
            }
        
        # Find features that are highly important but improve accuracy when removed
        spurious_candidates = [
            feat for feat, result in ablation_results.items()
            if result['is_spurious_candidate'] and result['importance'] > 0.1
        ]
        
        spurious_score = len(spurious_candidates) / X_test.shape[1]
        
        evidence = {
            'num_spurious_candidates': len(spurious_candidates),
            'spurious_features': spurious_candidates,
            'ablation_results': ablation_results,
            'baseline_accuracy': float(baseline_acc * 100)
        }
        
        return spurious_score, evidence
    
    def _compute_feature_importance(self, X, y):
        """Compute feature importance using gradients."""
        X_t = torch.FloatTensor(X).to(self.device).requires_grad_(True)
        y_t = torch.LongTensor(y).to(self.device)
        
        self.model.eval()
        output = self.model(X_t)
        loss = F.cross_entropy(output, y_t)
        loss.backward()
        
        # Gradient magnitude as importance
        importance = X_t.grad.abs().mean(dim=0).cpu().numpy()
        
        # Normalize
        importance = importance / (importance.sum() + 1e-10)
        
        return importance


class MissingFeatureAnalyzer:
    """Analyzes if failures could be fixed by additional features."""
    
    def analyze(self, X_train, y_train, X_test, y_test, failures):
        """
        Generate synthetic features and test if they help.
        
        Strategy: Create interaction terms, polynomial features, etc.
        """
        if failures.sum() < 10:
            return 0.0, {'reason': 'too_few_failures'}
        
        # Generate candidate features
        candidate_features = []
        
        # Interaction terms
        for i in range(min(3, X_train.shape[1])):
            for j in range(i+1, min(3, X_train.shape[1])):
                interaction = X_train[:, i] * X_train[:, j]
                candidate_features.append({
                    'name': f'f{i}_x_f{j}',
                    'train': interaction,
                    'test': X_test[:, i] * X_test[:, j]
                })
        
        # Polynomial features (squared)
        for i in range(min(3, X_train.shape[1])):
            squared = X_train[:, i] ** 2
            candidate_features.append({
                'name': f'f{i}_squared',
                'train': squared,
                'test': X_test[:, i] ** 2
            })
        
        # Test each candidate feature
        from sklearn.linear_model import LogisticRegression
        
        # Baseline model
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_acc = baseline_model.score(X_test[failures], y_test[failures])
        
        feature_scores = {}
        
        for candidate in candidate_features:
            # Add feature
            X_train_aug = np.column_stack([X_train, candidate['train']])
            X_test_aug = np.column_stack([X_test, candidate['test']])
            
            # Train model with new feature
            aug_model = LogisticRegression(max_iter=1000, random_state=42)
            aug_model.fit(X_train_aug, y_train)
            aug_acc = aug_model.score(X_test_aug[failures], y_test[failures])
            
            improvement = aug_acc - baseline_acc
            
            feature_scores[candidate['name']] = {
                'baseline_acc': float(baseline_acc * 100),
                'augmented_acc': float(aug_acc * 100),
                'improvement': float(improvement * 100),
                'is_helpful': improvement > 0.05
            }
        
        # Score based on how many features help
        helpful_features = [f for f, s in feature_scores.items() if s['is_helpful']]
        missing_feature_score = min(1.0, len(helpful_features) / 3)
        
        evidence = {
            'num_helpful_features': len(helpful_features),
            'helpful_features': helpful_features,
            'feature_scores': feature_scores,
            'baseline_acc_on_failures': float(baseline_acc * 100)
        }
        
        return missing_feature_score, evidence


class RootCauseAnalyzer:
    """Main analyzer that coordinates all diagnostic tests."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, device='cpu'):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        
        self.shift_detector = DataShiftDetector()
        self.noise_detector = LabelNoiseDetector(model, device)
        self.capacity_tester = ModelCapacityTester(device)
        self.spurious_detector = SpuriousCorrelationDetector(model, device)
        self.missing_feature_analyzer = MissingFeatureAnalyzer()
        
    def analyze(self, failures, cluster_indices=None):
        """
        Run all diagnostic tests and determine root cause.
        
        Args:
            failures: Boolean array of failures
            cluster_indices: Optional, analyze specific cluster
        """
        if cluster_indices is not None:
            # Focus on specific cluster
            failures = np.zeros_like(failures, dtype=bool)
            failures[cluster_indices] = True
        
        print("\n" + "="*70)
        print("ROOT CAUSE ANALYSIS")
        print("="*70)
        
        # Run all tests
        causes = {}
        
        print("\n[1/5] Testing for data shift...")
        shift_score, shift_evidence = self.shift_detector.detect(
            self.X_train, self.X_test, failures
        )
        causes['data_shift'] = {'score': shift_score, 'evidence': shift_evidence}
        print(f"  Data shift score: {shift_score:.3f}")
        
        print("\n[2/5] Testing for label noise...")
        noise_score, noise_evidence = self.noise_detector.detect(
            self.X_train, self.y_train, self.X_test, self.y_test, failures
        )
        causes['label_noise'] = {'score': noise_score, 'evidence': noise_evidence}
        print(f"  Label noise score: {noise_score:.3f}")
        
        print("\n[3/5] Testing model capacity...")
        capacity_score, capacity_evidence = self.capacity_tester.test(
            self.X_train, self.y_train, self.X_test, self.y_test, failures, self.model
        )
        causes['model_capacity'] = {'score': capacity_score, 'evidence': capacity_evidence}
        print(f"  Capacity issue score: {capacity_score:.3f}")
        print(f"  Diagnosis: {capacity_evidence.get('diagnosis', 'unknown')}")
        
        print("\n[4/5] Testing for spurious correlations...")
        spurious_score, spurious_evidence = self.spurious_detector.detect(
            self.X_train, self.y_train, self.X_test, self.y_test, failures
        )
        causes['spurious_correlation'] = {'score': spurious_score, 'evidence': spurious_evidence}
        print(f"  Spurious correlation score: {spurious_score:.3f}")
        
        print("\n[5/5] Analyzing for missing features...")
        missing_score, missing_evidence = self.missing_feature_analyzer.analyze(
            self.X_train, self.y_train, self.X_test, self.y_test, failures
        )
        causes['missing_feature'] = {'score': missing_score, 'evidence': missing_evidence}
        print(f"  Missing feature score: {missing_score:.3f}")
        
        # Determine primary root cause
        primary_cause = max(causes.items(), key=lambda x: x[1]['score'])
        
        print("\n" + "="*70)
        print(f"PRIMARY ROOT CAUSE: {primary_cause[0].upper()}")
        print(f"Confidence: {primary_cause[1]['score']:.3f}")
        print("="*70)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(primary_cause[0], primary_cause[1])
        
        return {
            'causes': causes,
            'primary_cause': primary_cause[0],
            'confidence': primary_cause[1]['score'],
            'recommendation': recommendation,
            'affected_samples': np.where(failures)[0]
        }
    
    def _generate_recommendation(self, cause_type, cause_data):
        """Generate actionable recommendation based on diagnosis."""
        recommendations = {
            'data_shift': "Collect more data from the shifted distribution or use domain adaptation techniques.",
            'label_noise': "Review and correct training labels, especially high-loss samples. Consider noise-robust loss functions.",
            'model_capacity': f"{'Increase model capacity (more layers/units)' if cause_data['evidence'].get('diagnosis') == 'underfitting' else 'Add regularization or reduce model size'}.",
            'spurious_correlation': "Remove or mask spurious features. Consider causal regularization.",
            'missing_feature': "Engineer new features, especially interaction terms and polynomial features."
        }
        
        return recommendations.get(cause_type, "Further investigation needed.")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("\nRoot Cause Analysis - Testing on Discovered Failures")
    print("="*70)
    
    # Load discovery results
    with open('./checkpoints/discovery_results.pkl', 'rb') as f:
        discovery_results = pickle.load(f)
    
    # Load models and data
    from failure_discovery import load_model_and_data
    
    failure_modes = ['label_noise', 'missing_feature', 'spurious', 'data_shift']
    
    diagnosis_results = {}
    
    for failure_mode in failure_modes:
        print(f"\n\n{'='*70}")
        print(f"Analyzing: {failure_mode.upper()}")
        print('='*70)
        
        try:
            # Load
            checkpoint_path = f'./checkpoints/model_{failure_mode}.pt'
            data_path = f'./checkpoints/data_{failure_mode}.pkl'
            
            model, data, checkpoint = load_model_and_data(checkpoint_path, data_path)
            
            # Analyze
            analyzer = RootCauseAnalyzer(
                model=model,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_test=data['X_test'],
                y_test=data['y_test']
            )
            
            failures = discovery_results[failure_mode]['metadata']['failures']
            diagnosis = analyzer.analyze(failures)
            
            diagnosis_results[failure_mode] = diagnosis
            
        except Exception as e:
            print(f"Error analyzing {failure_mode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save
    output_path = './checkpoints/diagnosis_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(diagnosis_results, f)
    
    print(f"\n\n{'='*70}")
    print("✓ Root cause analysis complete!")
    print(f"✓ Results saved to {output_path}")
    print('='*70)