"""
Module 4: Fix Validation Engine
Tests fixes in sandbox, validates statistically, and promotes only if better.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pickle
from dataclasses import dataclass
from pathlib import Path
import copy
from train import MLP, Trainer
from fix_proposer import FixGenerator, Fix
from failure_discovery import FailureCluster


@dataclass
class ValidationResult:
    """Results from validating a fix."""
    fix_id: str
    passed: bool
    baseline_metrics: Dict
    fixed_metrics: Dict
    improvement: Dict
    statistical_tests: Dict
    recommendation: str
    details: Dict


class Sandbox:
    """Isolated environment for testing fixes."""
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                 baseline_model, device='cpu'):
        self.X_train_orig = X_train.copy()
        self.y_train_orig = y_train.copy()
        self.X_val = X_val.copy()
        self.y_val = y_val.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.baseline_model = baseline_model
        self.device = device
        
    def apply_fix(self, fix: Fix, input_dim, num_classes):
        """
        Apply a fix and return modified data/model.
        
        Returns:
            (X_train_fixed, y_train_fixed, fixed_model, metadata)
        """
        print(f"\n  Applying fix: {fix.fix_id}")
        
        X_train = self.X_train_orig.copy()
        y_train = self.y_train_orig.copy()
        X_val = self.X_val.copy()
        X_test = self.X_test.copy()
        model = None
        sample_weights = None
        loss_fn = None
        metadata = {}
        
        # Apply fix based on type
        if fix.fix_type == 'data':
            if 'reweight' in fix.fix_id:
                X_train, y_train, sample_weights = fix.implementation(
                    X_train, y_train, X_test, fix.parameters
                )
                metadata['sample_weights'] = sample_weights
                
            elif 'remove' in fix.fix_id:
                X_train, y_train, _ = fix.implementation(
                    X_train, y_train, self.baseline_model, fix.parameters
                )
                
            elif 'augment' in fix.fix_id:
                X_train, y_train = fix.implementation(
                    X_train, y_train, fix.parameters
                )
                
            # Use baseline model architecture
            model = MLP(input_dim=X_train.shape[1], num_classes=num_classes)
            
        elif fix.fix_type == 'feature':
            if 'remove' in fix.fix_id:
                X_train = fix.implementation(X_train, fix.parameters)
                X_val = fix.implementation(X_val, fix.parameters)
                X_test = fix.implementation(X_test, fix.parameters)
                
            elif 'add' in fix.fix_id:
                X_train = fix.implementation(X_train, fix.parameters)
                X_val = fix.implementation(X_val, fix.parameters)
                X_test = fix.implementation(X_test, fix.parameters)
            
            # Model with new input dimension
            model = MLP(input_dim=X_train.shape[1], num_classes=num_classes)
            
        elif fix.fix_type == 'model':
            if 'capacity' in fix.fix_id:
                if 'increase' in fix.fix_id or 'reduce' in fix.fix_id:
                    model = fix.implementation(X_train.shape[1], num_classes, fix.parameters)
                elif 'ensemble' in fix.fix_id:
                    models = fix.implementation(X_train.shape[1], num_classes, fix.parameters)
                    model = models  # List of models
                elif 'regularize' in fix.fix_id:
                    model = MLP(input_dim=X_train.shape[1], num_classes=num_classes)
                    metadata['training_params'] = fix.implementation(fix.parameters)
                    
        elif fix.fix_type == 'loss':
            model = MLP(input_dim=X_train.shape[1], num_classes=num_classes)
            loss_fn = fix.implementation(fix.parameters)
            metadata['custom_loss'] = True
        
        # Default model if not set
        if model is None:
            model = MLP(input_dim=X_train.shape[1], num_classes=num_classes)
        
        metadata.update({
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_val_shape': X_val.shape,
            'X_test_shape': X_test.shape
        })
        
        return X_train, y_train, X_val, X_test, model, sample_weights, loss_fn, metadata
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   sample_weights=None, loss_fn=None, training_params=None):
        """Train model in sandbox."""
        print(f"    Training model...")
        
        # Prepare data loaders
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        if sample_weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        else:
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=128)
        
        # Training parameters
        if training_params is None:
            training_params = {}
        
        epochs = training_params.get('epochs', 15)
        lr = training_params.get('lr', 0.001)
        weight_decay = training_params.get('weight_decay', 0.0)
        
        # Train
        trainer = Trainer(model, device=self.device)
        
        # Custom loss if provided
        if loss_fn is not None:
            # Would need to modify Trainer to accept custom loss
            # For now, use default
            pass
        
        history = trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr)
        
        return model, history


class StatisticalValidator:
    """Performs statistical tests to validate improvements."""
    
    def __init__(self, significance_level=0.05):
        self.alpha = significance_level
        
    def validate(self, baseline_preds, baseline_labels, 
                fixed_preds, fixed_labels,
                test_indices=None):
        """
        Run statistical tests to determine if fixed model is better.
        
        Returns:
            Dict with test results
        """
        results = {}
        
        # 1. Paired t-test on sample-wise accuracy
        baseline_correct = (baseline_preds == baseline_labels).astype(float)
        fixed_correct = (fixed_preds == fixed_labels).astype(float)
        
        t_stat, p_value = stats.ttest_rel(fixed_correct, baseline_correct)
        
        results['paired_t_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha and t_stat > 0,
            'interpretation': 'Fixed model significantly better' if (p_value < self.alpha and t_stat > 0)
                            else 'No significant improvement'
        }
        
        # 2. McNemar's test (for binary predictions)
        contingency_table = np.array([
            [((baseline_preds == baseline_labels) & (fixed_preds == fixed_labels)).sum(),
             ((baseline_preds == baseline_labels) & (fixed_preds != fixed_labels)).sum()],
            [((baseline_preds != baseline_labels) & (fixed_preds == fixed_labels)).sum(),
             ((baseline_preds != baseline_labels) & (fixed_preds != fixed_labels)).sum()]
        ])
        
        # McNemar statistic
        b = contingency_table[0, 1]
        c = contingency_table[1, 0]
        
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            
            results['mcnemar_test'] = {
                'statistic': float(mcnemar_stat),
                'p_value': float(mcnemar_p),
                'significant': mcnemar_p < self.alpha and c > b,
                'b_c_difference': int(c - b)  # Positive means fixed is better
            }
        
        # 3. Bootstrap confidence intervals
        n_bootstrap = 1000
        baseline_accs = []
        fixed_accs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(baseline_labels), len(baseline_labels), replace=True)
            baseline_accs.append((baseline_preds[indices] == baseline_labels[indices]).mean())
            fixed_accs.append((fixed_preds[indices] == fixed_labels[indices]).mean())
        
        improvement = np.array(fixed_accs) - np.array(baseline_accs)
        ci_lower, ci_upper = np.percentile(improvement, [2.5, 97.5])
        
        results['bootstrap_ci'] = {
            'improvement_mean': float(improvement.mean()),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'significant': ci_lower > 0  # Entire CI above 0
        }
        
        # Overall decision
        num_significant = sum([
            results['paired_t_test']['significant'],
            results.get('mcnemar_test', {}).get('significant', False),
            results['bootstrap_ci']['significant']
        ])
        
        results['overall_significant'] = num_significant >= 2  # Majority vote
        
        return results


class FairnessChecker:
    """Checks if fix maintains fairness constraints."""
    
    def check(self, y_true, y_pred_baseline, y_pred_fixed, sensitive_feature=None):
        """
        Check fairness metrics.
        
        For now, just checks overall performance doesn't degrade on subgroups.
        """
        results = {}
        
        # Overall metrics
        baseline_acc = accuracy_score(y_true, y_pred_baseline)
        fixed_acc = accuracy_score(y_true, y_pred_fixed)
        
        results['overall'] = {
            'baseline_accuracy': float(baseline_acc),
            'fixed_accuracy': float(fixed_acc),
            'no_degradation': fixed_acc >= baseline_acc * 0.95  # Allow 5% tolerance
        }
        
        # If sensitive feature provided, check subgroup performance
        if sensitive_feature is not None:
            for group in np.unique(sensitive_feature):
                mask = sensitive_feature == group
                baseline_group_acc = accuracy_score(y_true[mask], y_pred_baseline[mask])
                fixed_group_acc = accuracy_score(y_true[mask], y_pred_fixed[mask])
                
                results[f'group_{group}'] = {
                    'baseline_accuracy': float(baseline_group_acc),
                    'fixed_accuracy': float(fixed_group_acc),
                    'no_degradation': fixed_group_acc >= baseline_group_acc * 0.95
                }
        
        results['fairness_maintained'] = all(
            v.get('no_degradation', True) for v in results.values() if isinstance(v, dict)
        )
        
        return results


class PromotionManager:
    """Decides whether to promote fix to production."""
    
    def __init__(self, strict_mode=True):
        self.strict_mode = strict_mode
        
    def should_promote(self, validation_result: ValidationResult) -> Tuple[bool, str]:
        """
        Decide if fix should be promoted.
        
        Returns:
            (should_promote: bool, reason: str)
        """
        checks = []
        
        # 1. Statistical significance
        if validation_result.statistical_tests['overall_significant']:
            checks.append(('statistical', True, 'Statistically significant improvement'))
        else:
            checks.append(('statistical', False, 'Not statistically significant'))
        
        # 2. Practical improvement (at least 2% better)
        improvement = validation_result.improvement['accuracy']
        if improvement > 2.0:  # 2% absolute improvement
            checks.append(('practical', True, f'{improvement:.1f}% improvement'))
        else:
            checks.append(('practical', False, f'Only {improvement:.1f}% improvement'))
        
        # 3. No regression on overall accuracy
        overall_safe = validation_result.fixed_metrics['accuracy'] >= \
                      validation_result.baseline_metrics['accuracy'] * 0.98
        checks.append(('safety', overall_safe, 
                      'No regression' if overall_safe else 'Performance degraded'))
        
        # 4. Fairness maintained
        fairness_ok = validation_result.details.get('fairness', {}).get('fairness_maintained', True)
        checks.append(('fairness', fairness_ok,
                      'Fairness maintained' if fairness_ok else 'Fairness violated'))
        
        # Decision logic
        if self.strict_mode:
            # All checks must pass
            should_promote = all(check[1] for check in checks)
            reason = ' | '.join(f"{check[0]}: {check[2]}" for check in checks)
        else:
            # Statistical significance + no major regression
            should_promote = checks[0][1] and checks[2][1] and checks[3][1]
            reason = ' | '.join(f"{check[0]}: {check[2]}" for check in checks)
        
        return should_promote, reason


class ValidationEngine:
    """Main validation engine."""
    
    def __init__(self, baseline_model, X_train, y_train, X_test, y_test, device='cpu'):
        # Split test into val and test
        X_val, X_test_final, y_val, y_test_final = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        self.sandbox = Sandbox(
            X_train, y_train, X_val, y_val, X_test_final, y_test_final,
            baseline_model, device
        )
        self.validator = StatisticalValidator()
        self.fairness_checker = FairnessChecker()
        self.promotion_manager = PromotionManager(strict_mode=False)
        
        # Get baseline metrics
        self.baseline_metrics = self._evaluate_model(
            baseline_model, X_test_final, y_test_final
        )
    
    def validate_fix(self, fix: Fix, input_dim, num_classes) -> ValidationResult:
        """
        Validate a single fix.
        
        Returns:
            ValidationResult with all test results
        """
        print(f"\n{'='*70}")
        print(f"Validating Fix: {fix.fix_id}")
        print('='*70)
        
        try:
            # Apply fix in sandbox
            X_train, y_train, X_val, y_val, X_test, model, sample_weights, loss_fn, metadata = \
                self.sandbox.apply_fix(fix, input_dim, num_classes)
            
            # Train fixed model
            training_params = metadata.get('training_params', {})
            fixed_model, history = self.sandbox.train_model(
                model, X_train, y_train, X_val, y_val,
                sample_weights, loss_fn, training_params
            )
            
            # Evaluate fixed model
            fixed_metrics = self._evaluate_model(fixed_model, X_test, self.sandbox.y_test)
            
            print(f"\n  Results:")
            print(f"    Baseline accuracy: {self.baseline_metrics['accuracy']:.2f}%")
            print(f"    Fixed accuracy: {fixed_metrics['accuracy']:.2f}%")
            print(f"    Improvement: {fixed_metrics['accuracy'] - self.baseline_metrics['accuracy']:.2f}%")
            
            # Statistical tests
            baseline_preds = self._get_predictions(
                self.sandbox.baseline_model, X_test, self.sandbox.y_test
            )
            fixed_preds = self._get_predictions(fixed_model, X_test, self.sandbox.y_test)
            
            stat_tests = self.validator.validate(
                baseline_preds, self.sandbox.y_test,
                fixed_preds, self.sandbox.y_test
            )
            
            print(f"\n  Statistical Tests:")
            print(f"    T-test significant: {stat_tests['paired_t_test']['significant']}")
            print(f"    Bootstrap CI significant: {stat_tests['bootstrap_ci']['significant']}")
            print(f"    Overall significant: {stat_tests['overall_significant']}")
            
            # Fairness check
            fairness = self.fairness_checker.check(
                self.sandbox.y_test, baseline_preds, fixed_preds
            )
            
            # Compute improvements
            improvement = {
                metric: fixed_metrics[metric] - self.baseline_metrics[metric]
                for metric in self.baseline_metrics.keys()
            }
            
            # Create result
            result = ValidationResult(
                fix_id=fix.fix_id,
                passed=stat_tests['overall_significant'] and fairness['fairness_maintained'],
                baseline_metrics=self.baseline_metrics,
                fixed_metrics=fixed_metrics,
                improvement=improvement,
                statistical_tests=stat_tests,
                recommendation='',
                details={
                    'fairness': fairness,
                    'training_history': history,
                    'metadata': metadata
                }
            )
            
            # Promotion decision
            should_promote, reason = self.promotion_manager.should_promote(result)
            result.recommendation = 'PROMOTE' if should_promote else 'REJECT'
            
            print(f"\n  Decision: {result.recommendation}")
            print(f"  Reason: {reason}")
            
            return result
            
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            return ValidationResult(
                fix_id=fix.fix_id,
                passed=False,
                baseline_metrics=self.baseline_metrics,
                fixed_metrics=self.baseline_metrics,
                improvement={k: 0 for k in self.baseline_metrics.keys()},
                statistical_tests={},
                recommendation='REJECT',
                details={'error': str(e)}
            )
    
    def _evaluate_model(self, model, X, y):
        """Evaluate model and return metrics."""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.sandbox.device)
        
        with torch.no_grad():
            output = model(X_tensor)
            preds = output.argmax(dim=1).cpu().numpy()
        
        return {
            'accuracy': float(accuracy_score(y, preds) * 100),
            'precision': float(precision_score(y, preds, average='weighted', zero_division=0) * 100),
            'recall': float(recall_score(y, preds, average='weighted', zero_division=0) * 100),
            'f1': float(f1_score(y, preds, average='weighted', zero_division=0) * 100)
        }
    
    def _get_predictions(self, model, X, y):
        """Get model predictions."""
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.sandbox.device)
        
        with torch.no_grad():
            output = model(X_tensor)
            preds = output.argmax(dim=1).cpu().numpy()
        
        return preds


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("\nFix Validation - Testing Proposed Fixes")
    print("="*70)
    
    # Load everything
    with open('./checkpoints/proposed_fixes.pkl', 'rb') as f:
        all_fixes = pickle.load(f)
    
    from failure_discovery import load_model_and_data
    
    failure_modes = ['label_noise', 'missing_feature', 'spurious', 'data_shift']
    
    validation_results = {}
    
    for failure_mode in failure_modes:
        print(f"\n\n{'='*70}")
        print(f"Validating fixes for: {failure_mode.upper()}")
        print('='*70)
        
        try:
            # Load model and data
            checkpoint_path = f'./checkpoints/model_{failure_mode}.pt'
            data_path = f'./checkpoints/data_{failure_mode}.pkl'
            
            model, data, checkpoint = load_model_and_data(checkpoint_path, data_path)
            
            # Create validation engine
            engine = ValidationEngine(
                baseline_model=model,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_test=data['X_test'],
                y_test=data['y_test']
            )
            
            # Validate each fix (top 2 only for speed)
            fixes = all_fixes[failure_mode][:2]
            results = []
            
            for fix in fixes:
                result = engine.validate_fix(
                    fix,
                    input_dim=checkpoint['input_dim'],
                    num_classes=checkpoint['num_classes']
                )
                results.append(result)
            
            validation_results[failure_mode] = results
            
            # Summary
            promoted = sum(1 for r in results if r.recommendation == 'PROMOTE')
            print(f"\n  Summary: {promoted}/{len(results)} fixes promoted")
            
        except Exception as e:
            print(f"Error validating {failure_mode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save
    output_path = './checkpoints/validation_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(validation_results, f)
    
    print(f"\n\n{'='*70}")
    print("✓ Fix validation complete!")
    print(f"✓ Results saved to {output_path}")
    print('='*70)