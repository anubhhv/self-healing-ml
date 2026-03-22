"""
Module 3: Automatic Fix Proposal Engine
Maps root causes to concrete fixes with implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import pickle


@dataclass
class Fix:
    """Represents a proposed fix."""
    fix_id: str
    fix_type: str  # 'data', 'feature', 'model', 'loss'
    description: str
    implementation: Callable  # Function to apply the fix
    expected_impact: float  # Estimated error reduction (0-1)
    cost: Dict  # {'compute': ..., 'data_collection': ..., 'time': ...}
    parameters: Dict  # Hyperparameters for the fix
    priority: int  # 1 (high) to 5 (low)


class FixGenerator:
    """Generates fixes based on diagnosed root causes."""
    
    def __init__(self):
        self.fix_strategies = {
            'data_shift': self._generate_data_shift_fixes,
            'label_noise': self._generate_label_noise_fixes,
            'model_capacity': self._generate_capacity_fixes,
            'spurious_correlation': self._generate_spurious_fixes,
            'missing_feature': self._generate_missing_feature_fixes
        }
    
    def generate(self, diagnosis: Dict) -> List[Fix]:
        """
        Generate fixes based on diagnosis.
        
        Args:
            diagnosis: Output from RootCauseAnalyzer
        
        Returns:
            List of Fix objects, ranked by priority
        """
        primary_cause = diagnosis['primary_cause']
        confidence = diagnosis['confidence']
        evidence = diagnosis['causes'][primary_cause]['evidence']
        
        # Get appropriate fix generation strategy
        if primary_cause in self.fix_strategies:
            fixes = self.fix_strategies[primary_cause](evidence, confidence)
        else:
            fixes = []
        
        # Rank fixes by expected impact / cost
        fixes = self._rank_fixes(fixes)
        
        return fixes
    
    def _generate_data_shift_fixes(self, evidence: Dict, confidence: float) -> List[Fix]:
        """Generate fixes for data shift."""
        fixes = []
        
        # Fix 1: Importance reweighting
        fixes.append(Fix(
            fix_id='data_shift_reweight',
            fix_type='data',
            description='Reweight training samples using importance sampling to match test distribution',
            implementation=self._impl_importance_reweighting,
            expected_impact=confidence * 0.7,
            cost={'compute': 'low', 'data_collection': 'none', 'time': 'minutes'},
            parameters={
                'method': 'density_ratio',
                'bandwidth': 'auto'
            },
            priority=1
        ))
        
        # Fix 2: Collect more data from shifted region
        shifted_features = [
            feat for feat, data in evidence.get('feature_shifts', {}).items()
            if data['p_value'] < 0.05
        ]
        
        fixes.append(Fix(
            fix_id='data_shift_collect',
            fix_type='data',
            description=f'Collect more training data from shifted regions: {shifted_features}',
            implementation=self._impl_data_collection_strategy,
            expected_impact=confidence * 0.9,
            cost={'compute': 'none', 'data_collection': 'high', 'time': 'weeks'},
            parameters={
                'target_features': shifted_features,
                'sample_size': 500
            },
            priority=2
        ))
        
        # Fix 3: Domain adaptation
        fixes.append(Fix(
            fix_id='data_shift_adaptation',
            fix_type='model',
            description='Apply domain adaptation to align train and test distributions',
            implementation=self._impl_domain_adaptation,
            expected_impact=confidence * 0.6,
            cost={'compute': 'medium', 'data_collection': 'none', 'time': 'hours'},
            parameters={
                'method': 'adversarial',
                'lambda_domain': 0.1
            },
            priority=3
        ))
        
        return fixes
    
    def _generate_label_noise_fixes(self, evidence: Dict, confidence: float) -> List[Fix]:
        """Generate fixes for label noise."""
        fixes = []
        
        # Fix 1: Remove high-loss samples
        fixes.append(Fix(
            fix_id='noise_remove_samples',
            fix_type='data',
            description='Remove training samples with consistently high loss (likely mislabeled)',
            implementation=self._impl_remove_noisy_samples,
            expected_impact=confidence * 0.8,
            cost={'compute': 'low', 'data_collection': 'none', 'time': 'minutes'},
            parameters={
                'loss_percentile': 95,
                'cross_val_folds': 3
            },
            priority=1
        ))
        
        # Fix 2: Noise-robust loss function
        fixes.append(Fix(
            fix_id='noise_robust_loss',
            fix_type='loss',
            description='Use noise-robust loss function (symmetric cross-entropy)',
            implementation=self._impl_robust_loss,
            expected_impact=confidence * 0.7,
            cost={'compute': 'low', 'data_collection': 'none', 'time': 'minutes'},
            parameters={
                'loss_type': 'symmetric_ce',
                'alpha': 0.1,
                'beta': 1.0
            },
            priority=2
        ))
        
        # Fix 3: Relabel suspicious samples
        suspicious_count = evidence.get('num_suspicious_train_samples', 0)
        
        fixes.append(Fix(
            fix_id='noise_relabel',
            fix_type='data',
            description=f'Manually review and relabel {suspicious_count} suspicious training samples',
            implementation=self._impl_relabel_strategy,
            expected_impact=confidence * 0.9,
            cost={'compute': 'none', 'data_collection': 'medium', 'time': 'days'},
            parameters={
                'num_samples': suspicious_count,
                'selection_method': 'high_loss'
            },
            priority=3
        ))
        
        return fixes
    
    def _generate_capacity_fixes(self, evidence: Dict, confidence: float) -> List[Fix]:
        """Generate fixes for model capacity issues."""
        fixes = []
        
        diagnosis = evidence.get('diagnosis', 'unknown')
        
        if diagnosis == 'underfitting':
            # Fix 1: Increase model size
            fixes.append(Fix(
                fix_id='capacity_increase',
                fix_type='model',
                description='Increase model capacity (deeper/wider network)',
                implementation=self._impl_increase_capacity,
                expected_impact=confidence * 0.8,
                cost={'compute': 'high', 'data_collection': 'none', 'time': 'hours'},
                parameters={
                    'hidden_dims': [128, 64, 32, 16],
                    'dropout': 0.2
                },
                priority=1
            ))
            
            # Fix 2: Use ensemble
            fixes.append(Fix(
                fix_id='capacity_ensemble',
                fix_type='model',
                description='Use ensemble of models to increase effective capacity',
                implementation=self._impl_ensemble,
                expected_impact=confidence * 0.7,
                cost={'compute': 'very_high', 'data_collection': 'none', 'time': 'hours'},
                parameters={
                    'num_models': 5,
                    'diversity_method': 'bagging'
                },
                priority=2
            ))
            
        elif diagnosis == 'overfitting':
            # Fix 1: Add regularization
            fixes.append(Fix(
                fix_id='capacity_regularize',
                fix_type='model',
                description='Add L2 regularization and increase dropout',
                implementation=self._impl_add_regularization,
                expected_impact=confidence * 0.7,
                cost={'compute': 'medium', 'data_collection': 'none', 'time': 'minutes'},
                parameters={
                    'weight_decay': 0.01,
                    'dropout': 0.5
                },
                priority=1
            ))
            
            # Fix 2: Reduce model size
            fixes.append(Fix(
                fix_id='capacity_reduce',
                fix_type='model',
                description='Reduce model capacity to prevent overfitting',
                implementation=self._impl_reduce_capacity,
                expected_impact=confidence * 0.6,
                cost={'compute': 'low', 'data_collection': 'none', 'time': 'minutes'},
                parameters={
                    'hidden_dims': [32, 16]
                },
                priority=2
            ))
            
            # Fix 3: Collect more data
            fixes.append(Fix(
                fix_id='capacity_more_data',
                fix_type='data',
                description='Collect more training data to prevent overfitting',
                implementation=self._impl_collect_more_data,
                expected_impact=confidence * 0.8,
                cost={'compute': 'none', 'data_collection': 'high', 'time': 'weeks'},
                parameters={
                    'target_size': 2000
                },
                priority=3
            ))
        
        return fixes
    
    def _generate_spurious_fixes(self, evidence: Dict, confidence: float) -> List[Fix]:
        """Generate fixes for spurious correlations."""
        fixes = []
        
        spurious_features = evidence.get('spurious_features', [])
        
        # Fix 1: Remove spurious features
        if spurious_features:
            fixes.append(Fix(
                fix_id='spurious_remove',
                fix_type='feature',
                description=f'Remove spurious features: {spurious_features}',
                implementation=self._impl_remove_features,
                expected_impact=confidence * 0.8,
                cost={'compute': 'low', 'data_collection': 'none', 'time': 'minutes'},
                parameters={
                    'features_to_remove': spurious_features
                },
                priority=1
            ))
        
        # Fix 2: Feature masking with regularization
        fixes.append(Fix(
            fix_id='spurious_mask',
            fix_type='model',
            description='Apply gradient masking to reduce reliance on spurious features',
            implementation=self._impl_feature_masking,
            expected_impact=confidence * 0.6,
            cost={'compute': 'medium', 'data_collection': 'none', 'time': 'hours'},
            parameters={
                'masking_rate': 0.5,
                'penalty_weight': 0.01
            },
            priority=2
        ))
        
        # Fix 3: Data augmentation to break correlation
        fixes.append(Fix(
            fix_id='spurious_augment',
            fix_type='data',
            description='Apply data augmentation to break spurious correlations',
            implementation=self._impl_decorrelation_augmentation,
            expected_impact=confidence * 0.7,
            cost={'compute': 'medium', 'data_collection': 'none', 'time': 'hours'},
            parameters={
                'augmentation_strength': 0.3
            },
            priority=3
        ))
        
        return fixes
    
    def _generate_missing_feature_fixes(self, evidence: Dict, confidence: float) -> List[Fix]:
        """Generate fixes for missing features."""
        fixes = []
        
        helpful_features = evidence.get('helpful_features', [])
        
        # Fix 1: Add interaction features
        fixes.append(Fix(
            fix_id='missing_add_interactions',
            fix_type='feature',
            description=f'Add interaction features: {helpful_features}',
            implementation=self._impl_add_features,
            expected_impact=confidence * 0.8,
            cost={'compute': 'low', 'data_collection': 'none', 'time': 'minutes'},
            parameters={
                'features_to_add': helpful_features,
                'method': 'interaction'
            },
            priority=1
        ))
        
        # Fix 2: Automated feature engineering
        fixes.append(Fix(
            fix_id='missing_auto_engineer',
            fix_type='feature',
            description='Apply automated feature engineering (polynomial, log transforms, etc.)',
            implementation=self._impl_auto_feature_engineering,
            expected_impact=confidence * 0.6,
            cost={'compute': 'medium', 'data_collection': 'none', 'time': 'hours'},
            parameters={
                'max_degree': 2,
                'include_transforms': ['log', 'sqrt', 'square']
            },
            priority=2
        ))
        
        # Fix 3: Collect new features
        fixes.append(Fix(
            fix_id='missing_collect_features',
            fix_type='data',
            description='Collect additional features based on domain knowledge',
            implementation=self._impl_feature_collection_strategy,
            expected_impact=confidence * 0.9,
            cost={'compute': 'none', 'data_collection': 'very_high', 'time': 'months'},
            parameters={
                'suggested_features': ['domain_specific']
            },
            priority=3
        ))
        
        return fixes
    
    def _rank_fixes(self, fixes: List[Fix]) -> List[Fix]:
        """Rank fixes by expected impact / cost ratio."""
        cost_map = {
            'none': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 4
        }
        
        for fix in fixes:
            # Compute composite cost
            compute_cost = cost_map.get(fix.cost.get('compute', 'medium'), 2)
            data_cost = cost_map.get(fix.cost.get('data_collection', 'medium'), 2)
            total_cost = compute_cost + data_cost * 2  # Data collection is more expensive
            
            # Score = impact / (cost + 1)
            fix.score = fix.expected_impact / (total_cost + 1)
        
        # Sort by score (high to low), then by priority
        fixes.sort(key=lambda x: (x.score, -x.priority), reverse=True)
        
        return fixes
    
    # ========================================================================
    # FIX IMPLEMENTATION FUNCTIONS
    # ========================================================================
    
    def _impl_importance_reweighting(self, X_train, y_train, X_test, params):
        """Reweight training samples to match test distribution."""
        from sklearn.neighbors import KernelDensity
        
        # Estimate densities
        train_density = KernelDensity(bandwidth=params.get('bandwidth', 'auto'))
        train_density.fit(X_train)
        
        test_density = KernelDensity(bandwidth=params.get('bandwidth', 'auto'))
        test_density.fit(X_test)
        
        # Compute importance weights: p_test(x) / p_train(x)
        train_log_density = train_density.score_samples(X_train)
        test_log_density = test_density.score_samples(X_train)
        
        weights = np.exp(test_log_density - train_log_density)
        weights = np.clip(weights, 0.1, 10)  # Clip extreme weights
        weights /= weights.sum()  # Normalize
        weights *= len(weights)  # Scale back to average of 1
        
        return X_train, y_train, weights
    
    def _impl_data_collection_strategy(self, X_train, y_train, X_test, params):
        """Strategy for collecting more data (returns query regions)."""
        target_features = params.get('target_features', [])
        sample_size = params.get('sample_size', 500)
        
        # Identify regions to sample from
        query_regions = {}
        for feat in target_features:
            feat_idx = int(feat.split('_')[1])
            feat_values = X_test[:, feat_idx]
            
            # Find high-density regions in test set
            low, high = np.percentile(feat_values, [25, 75])
            query_regions[feat] = (low, high)
        
        return {
            'query_regions': query_regions,
            'sample_size': sample_size,
            'message': f'Collect {sample_size} samples where: {query_regions}'
        }
    
    def _impl_domain_adaptation(self, model, X_train, y_train, X_test, params):
        """Simple domain adaptation (would need full implementation)."""
        # Placeholder for domain adaptation
        # Real implementation would use adversarial training
        return {
            'message': 'Domain adaptation requires adversarial training',
            'method': params.get('method'),
            'hyperparameters': params
        }
    
    def _impl_remove_noisy_samples(self, X_train, y_train, model, params):
        """Remove samples with consistently high loss."""
        from torch.utils.data import TensorDataset, DataLoader
        import torch.nn.functional as F
        
        percentile = params.get('loss_percentile', 95)
        
        # Compute losses
        model.eval()
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            losses = F.cross_entropy(outputs, y_tensor, reduction='none')
            losses = losses.numpy()
        
        # Remove high-loss samples
        threshold = np.percentile(losses, percentile)
        keep_mask = losses < threshold
        
        X_cleaned = X_train[keep_mask]
        y_cleaned = y_train[keep_mask]
        
        print(f"Removed {(~keep_mask).sum()} noisy samples ({(~keep_mask).mean()*100:.1f}%)")
        
        return X_cleaned, y_cleaned, None
    
    def _impl_robust_loss(self, params):
        """Return robust loss function."""
        loss_type = params.get('loss_type', 'symmetric_ce')
        alpha = params.get('alpha', 0.1)
        beta = params.get('beta', 1.0)
        
        if loss_type == 'symmetric_ce':
            def symmetric_cross_entropy(output, target):
                ce = F.cross_entropy(output, target)
                rce = F.cross_entropy(target.unsqueeze(0).float(), 
                                     torch.softmax(output, dim=1).transpose(0, 1))
                return alpha * ce + beta * rce
            
            return symmetric_cross_entropy
        
        return F.cross_entropy
    
    def _impl_relabel_strategy(self, X_train, y_train, model, params):
        """Strategy for relabeling samples."""
        num_samples = params.get('num_samples', 100)
        
        # Find high-loss samples
        model.eval()
        with torch.no_grad():
            outputs = model(torch.FloatTensor(X_train))
            losses = F.cross_entropy(outputs, torch.LongTensor(y_train), reduction='none')
            losses = losses.numpy()
        
        # Get indices of highest loss samples
        suspicious_indices = np.argsort(losses)[-num_samples:]
        
        return {
            'indices_to_relabel': suspicious_indices,
            'message': f'Please review and relabel samples at indices: {suspicious_indices[:10]}...'
        }
    
    def _impl_increase_capacity(self, input_dim, num_classes, params):
        """Create larger model."""
        from train import MLP
        
        hidden_dims = params.get('hidden_dims', [128, 64, 32])
        dropout = params.get('dropout', 0.2)
        
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, 
                   num_classes=num_classes, dropout=dropout)
        
        return model
    
    def _impl_ensemble(self, input_dim, num_classes, params):
        """Create ensemble of models."""
        from train import MLP
        
        num_models = params.get('num_models', 5)
        
        models = []
        for i in range(num_models):
            model = MLP(input_dim=input_dim, num_classes=num_classes)
            models.append(model)
        
        return models
    
    def _impl_add_regularization(self, params):
        """Return regularization parameters."""
        return {
            'weight_decay': params.get('weight_decay', 0.01),
            'dropout': params.get('dropout', 0.5)
        }
    
    def _impl_reduce_capacity(self, input_dim, num_classes, params):
        """Create smaller model."""
        from train import MLP
        
        hidden_dims = params.get('hidden_dims', [32, 16])
        
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
        
        return model
    
    def _impl_collect_more_data(self, params):
        """Strategy for collecting more data."""
        target_size = params.get('target_size', 2000)
        
        return {
            'message': f'Collect {target_size} more training samples',
            'target_size': target_size
        }
    
    def _impl_remove_features(self, X, params):
        """Remove specified features."""
        features_to_remove = params.get('features_to_remove', [])
        
        # Extract indices
        indices_to_remove = [int(f.split('_')[1]) for f in features_to_remove]
        
        # Remove features
        mask = np.ones(X.shape[1], dtype=bool)
        mask[indices_to_remove] = False
        
        X_filtered = X[:, mask]
        
        return X_filtered
    
    def _impl_feature_masking(self, params):
        """Return feature masking parameters."""
        return {
            'masking_rate': params.get('masking_rate', 0.5),
            'penalty_weight': params.get('penalty_weight', 0.01)
        }
    
    def _impl_decorrelation_augmentation(self, X, y, params):
        """Augment data to break correlations."""
        strength = params.get('augmentation_strength', 0.3)
        
        # Add noise to features
        noise = np.random.normal(0, strength, X.shape)
        X_augmented = X + noise
        
        return np.vstack([X, X_augmented]), np.hstack([y, y])
    
    def _impl_add_features(self, X, params):
        """Add new features."""
        features_to_add = params.get('features_to_add', [])
        method = params.get('method', 'interaction')
        
        new_features = []
        
        for feat_spec in features_to_add:
            if 'x' in feat_spec:  # Interaction
                parts = feat_spec.split('_x_')
                i = int(parts[0].replace('f', ''))
                j = int(parts[1].replace('f', ''))
                new_features.append(X[:, i] * X[:, j])
            elif 'squared' in feat_spec:
                i = int(feat_spec.split('_')[0].replace('f', ''))
                new_features.append(X[:, i] ** 2)
        
        if new_features:
            X_augmented = np.column_stack([X] + new_features)
            return X_augmented
        
        return X
    
    def _impl_auto_feature_engineering(self, X, params):
        """Automated feature engineering."""
        max_degree = params.get('max_degree', 2)
        
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=max_degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        return X_poly
    
    def _impl_feature_collection_strategy(self, params):
        """Strategy for collecting new features."""
        return {
            'message': 'Collect additional features based on domain expertise',
            'suggested_features': params.get('suggested_features', [])
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("\nFix Proposal - Generating Fixes for Diagnosed Root Causes")
    print("="*70)
    
    # Load diagnosis results
    with open('./checkpoints/diagnosis_results.pkl', 'rb') as f:
        diagnosis_results = pickle.load(f)
    
    generator = FixGenerator()
    
    all_fixes = {}
    
    for failure_mode, diagnosis in diagnosis_results.items():
        print(f"\n\n{'='*70}")
        print(f"Generating fixes for: {failure_mode.upper()}")
        print(f"Diagnosed cause: {diagnosis['primary_cause']}")
        print(f"Confidence: {diagnosis['confidence']:.3f}")
        print('='*70)
        
        fixes = generator.generate(diagnosis)
        
        print(f"\nGenerated {len(fixes)} fixes:")
        for i, fix in enumerate(fixes, 1):
            print(f"\n  Fix {i}: {fix.fix_id}")
            print(f"    Type: {fix.fix_type}")
            print(f"    Description: {fix.description}")
            print(f"    Expected impact: {fix.expected_impact:.3f}")
            print(f"    Cost: {fix.cost}")
            print(f"    Priority: {fix.priority}")
            print(f"    Score: {fix.score:.3f}")
        
        all_fixes[failure_mode] = fixes
    
    # Save
    output_path = './checkpoints/proposed_fixes.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(all_fixes, f)
    
    print(f"\n\n{'='*70}")
    print("✓ Fix proposal complete!")
    print(f"✓ Results saved to {output_path}")
    print('='*70)