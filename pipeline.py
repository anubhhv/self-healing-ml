"""
Self-Healing ML System - Main Pipeline
Orchestrates: Train → Deploy → Fail → Discover → Diagnose → Fix → Validate → Heal
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

# Import all modules
from train import train_model_with_failures, MLP
from failure_discovery import FailureDiscoveryEngine, load_model_and_data
from root_cause_analyzer import RootCauseAnalyzer
from fix_proposer import FixGenerator
from validation_engine import ValidationEngine

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

class SelfHealingPipeline:
    """Main pipeline that orchestrates the entire self-healing process."""
    
    def __init__(self, output_dir='./checkpoints', device='cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.results = {}
        
    def run_full_cycle(self, dataset_name='adult', failure_mode=None, 
                       failure_params=None, auto_heal=True):
        """
        Run complete self-healing cycle.
        
        Args:
            dataset_name: Dataset to use
            failure_mode: Type of failure to inject
            failure_params: Parameters for failure injection
            auto_heal: If True, automatically apply best fix
        
        Returns:
            Dict with complete results
        """
        print("\n" + "="*80)
        print("SELF-HEALING ML SYSTEM - FULL CYCLE")
        print("="*80)
        print(f"Dataset: {dataset_name}")
        print(f"Failure Mode: {failure_mode or 'clean'}")
        print(f"Auto-Heal: {auto_heal}")
        print("="*80)
        
        cycle_results = {
            'dataset': dataset_name,
            'failure_mode': failure_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        # ====================================================================
        # STAGE 1: TRAIN
        # ====================================================================
        print("\n" + "▶" * 40)
        print("STAGE 1: TRAINING MODEL")
        print("▶" * 40)
        
        checkpoint, model, trainer = train_model_with_failures(
            dataset_name=dataset_name,
            failure_mode=failure_mode,
            failure_params=failure_params,
            save_path=self.output_dir
        )
        
        cycle_results['training'] = {
            'final_val_acc': checkpoint['history']['val_acc'][-1],
            'final_train_acc': checkpoint['history']['train_acc'][-1],
            'checkpoint_path': str(self.output_dir / f"model_{failure_mode or 'clean'}.pt")
        }
        
        print(f"\n✓ Training complete. Val Acc: {checkpoint['history']['val_acc'][-1]:.2f}%")
        
        # ====================================================================
        # STAGE 2: DEPLOY & FAIL
        # ====================================================================
        print("\n" + "▶" * 40)
        print("STAGE 2: DEPLOYMENT & FAILURE DETECTION")
        print("▶" * 40)
        
        # Load data
        data_path = self.output_dir / f"data_{failure_mode or 'clean'}.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if there are failures
        model.eval()
        X_test_t = torch.FloatTensor(data['X_test']).to(self.device)
        with torch.no_grad():
            output = model(X_test_t)
            preds = output.argmax(dim=1).cpu().numpy()
        
        failures = preds != data['y_test']
        failure_rate = failures.mean()
        
        print(f"\n✓ Deployed. Failure Rate: {failure_rate*100:.2f}%")
        
        if failure_rate < 0.05:
            print("\n✓ Model performing well! No healing needed.")
            cycle_results['status'] = 'healthy'
            return cycle_results
        
        cycle_results['deployment'] = {
            'failure_rate': float(failure_rate),
            'num_failures': int(failures.sum())
        }
        
        # ====================================================================
        # STAGE 3: DISCOVER FAILURES
        # ====================================================================
        print("\n" + "▶" * 40)
        print("STAGE 3: FAILURE DISCOVERY")
        print("▶" * 40)
        
        discovery_engine = FailureDiscoveryEngine(
            model=model,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test']
        )
        
        discovery_results = discovery_engine.discover()
        cycle_results['discovery'] = discovery_results['summary']
        
        # Save discovery results
        discovery_path = self.output_dir / f"discovery_{failure_mode or 'clean'}.pkl"
        with open(discovery_path, 'wb') as f:
            pickle.dump(discovery_results, f)
        
        # ====================================================================
        # STAGE 4: DIAGNOSE ROOT CAUSE
        # ====================================================================
        print("\n" + "▶" * 40)
        print("STAGE 4: ROOT CAUSE ANALYSIS")
        print("▶" * 40)
        
        analyzer = RootCauseAnalyzer(
            model=model,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            device=self.device
        )
        
        diagnosis = analyzer.analyze(discovery_results['metadata']['failures'])
        
        cycle_results['diagnosis'] = {
            'primary_cause': diagnosis['primary_cause'],
            'confidence': diagnosis['confidence'],
            'recommendation': diagnosis['recommendation']
        }
        
        # Save diagnosis
        diagnosis_path = self.output_dir / f"diagnosis_{failure_mode or 'clean'}.pkl"
        with open(diagnosis_path, 'wb') as f:
            pickle.dump(diagnosis, f)
        
        # ====================================================================
        # STAGE 5: PROPOSE FIXES
        # ====================================================================
        print("\n" + "▶" * 40)
        print("STAGE 5: FIX PROPOSAL")
        print("▶" * 40)
        
        fix_generator = FixGenerator()
        proposed_fixes = fix_generator.generate(diagnosis)
        
        print(f"\n✓ Generated {len(proposed_fixes)} fixes")
        print("\nTop 3 Fixes:")
        for i, fix in enumerate(proposed_fixes[:3], 1):
            print(f"\n  {i}. {fix.fix_id}")
            print(f"     Description: {fix.description}")
            print(f"     Expected Impact: {fix.expected_impact:.3f}")
            print(f"     Cost: {fix.cost}")
            print(f"     Priority: {fix.priority}")
        
        cycle_results['proposed_fixes'] = [
            {
                'fix_id': fix.fix_id,
                'fix_type': fix.fix_type,
                'description': fix.description,
                'expected_impact': fix.expected_impact,
                'priority': fix.priority
            }
            for fix in proposed_fixes
        ]
        
        # Save fixes
        fixes_path = self.output_dir / f"fixes_{failure_mode or 'clean'}.pkl"
        with open(fixes_path, 'wb') as f:
            pickle.dump(proposed_fixes, f)
        
        # ====================================================================
        # STAGE 6: VALIDATE FIXES
        # ====================================================================
        if not auto_heal:
            print("\n✓ Auto-heal disabled. Stopping here.")
            cycle_results['status'] = 'fixes_proposed'
            return cycle_results
        
        print("\n" + "▶" * 40)
        print("STAGE 6: FIX VALIDATION")
        print("▶" * 40)
        
        validation_engine = ValidationEngine(
            baseline_model=model,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            device=self.device
        )
        
        validation_results = []
        best_fix = None
        best_improvement = 0
        
        # Validate top 3 fixes
        for fix in proposed_fixes[:3]:
            result = validation_engine.validate_fix(
                fix,
                input_dim=checkpoint['input_dim'],
                num_classes=checkpoint['num_classes']
            )
            validation_results.append(result)
            
            # Track best fix
            if result.recommendation == 'PROMOTE':
                improvement = result.improvement['accuracy']
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_fix = (fix, result)
        
        # Save validation results
        validation_path = self.output_dir / f"validation_{failure_mode or 'clean'}.pkl"
        with open(validation_path, 'wb') as f:
            pickle.dump(validation_results, f)
        
        cycle_results['validation'] = [
            {
                'fix_id': r.fix_id,
                'passed': r.passed,
                'recommendation': r.recommendation,
                'improvement': r.improvement['accuracy']
            }
            for r in validation_results
        ]
        
        # ====================================================================
        # STAGE 7: HEAL (DEPLOY BEST FIX)
        # ====================================================================
        if best_fix is not None:
            print("\n" + "▶" * 40)
            print("STAGE 7: HEALING")
            print("▶" * 40)
            
            fix, result = best_fix
            
            print(f"\n✓ Deploying fix: {fix.fix_id}")
            print(f"  Improvement: {result.improvement['accuracy']:.2f}%")
            print(f"  Baseline Acc: {result.baseline_metrics['accuracy']:.2f}%")
            print(f"  Fixed Acc: {result.fixed_metrics['accuracy']:.2f}%")
            
            cycle_results['healing'] = {
                'fix_applied': fix.fix_id,
                'baseline_accuracy': result.baseline_metrics['accuracy'],
                'healed_accuracy': result.fixed_metrics['accuracy'],
                'improvement': result.improvement['accuracy']
            }
            
            cycle_results['status'] = 'healed'
            
            print("\n" + "="*80)
            print("✓ SYSTEM SUCCESSFULLY HEALED!")
            print("="*80)
            
        else:
            print("\n✗ No fixes passed validation")
            cycle_results['status'] = 'no_valid_fixes'
        
        # ====================================================================
        # SAVE FINAL RESULTS
        # ====================================================================
        results_path = self.output_dir / f"cycle_results_{failure_mode or 'clean'}.json"
        with open(results_path, 'w') as f:
            json.dump(to_json_safe(cycle_results), f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
        return cycle_results
    
    def run_experiments(self):
        """Run experiments on all failure modes."""
        print("\n" + "="*80)
        print("RUNNING FULL EXPERIMENTAL SUITE")
        print("="*80)
        
        failure_modes = [
            ('clean', None, {}),
            ('label_noise', 'label_noise', {
                'condition': lambda x: x[0] > 0.5 and x[1] < 0,
                'noise_rate': 0.3
            }),
            ('missing_feature', 'missing_feature', {
                'feature_idx': 0
            }),
            ('spurious', 'spurious', {
                'correlation_strength': 0.9
            }),
            ('data_shift', 'data_shift', {
                'shift_type': 'covariate'
            })
        ]
        
        all_results = {}
        
        for name, mode, params in failure_modes:
            print(f"\n\n{'#'*80}")
            print(f"EXPERIMENT: {name.upper()}")
            print('#'*80)
            
            results = self.run_full_cycle(
                dataset_name='adult',
                failure_mode=mode,
                failure_params=params,
                auto_heal=True
            )
            
            all_results[name] = results
        
        # Summary
        print("\n\n" + "="*80)
        print("EXPERIMENTAL SUMMARY")
        print("="*80)
        
        for name, results in all_results.items():
            print(f"\n{name.upper()}:")
            print(f"  Status: {results.get('status', 'unknown')}")
            
            if 'training' in results:
                print(f"  Training Acc: {results['training']['final_val_acc']:.2f}%")
            
            if 'healing' in results:
                print(f"  Baseline Acc: {results['healing']['baseline_accuracy']:.2f}%")
                print(f"  Healed Acc: {results['healing']['healed_accuracy']:.2f}%")
                print(f"  Improvement: +{results['healing']['improvement']:.2f}%")
        
        # Save summary
        summary_path = self.output_dir / 'experimental_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_path}")
        
        return all_results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Healing ML System')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'experiments'],
                       help='Run single cycle or full experiments')
    parser.add_argument('--failure', type=str, default=None,
                       choices=['label_noise', 'missing_feature', 'spurious', 'data_shift'],
                       help='Failure mode to inject')
    parser.add_argument('--auto-heal', action='store_true',
                       help='Automatically apply best fix')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = SelfHealingPipeline(output_dir=args.output_dir)
    
    if args.mode == 'single':
        # Single cycle
        failure_params = None
        if args.failure == 'label_noise':
            failure_params = {
                'condition': lambda x: x[0] > 0.5 and x[1] < 0,
                'noise_rate': 0.3
            }
        elif args.failure == 'missing_feature':
            failure_params = {'feature_idx': 0}
        elif args.failure == 'spurious':
            failure_params = {'correlation_strength': 0.9}
        elif args.failure == 'data_shift':
            failure_params = {'shift_type': 'covariate'}
        
        results = pipeline.run_full_cycle(
            failure_mode=args.failure,
            failure_params=failure_params,
            auto_heal=args.auto_heal
        )
        
    else:
        # Full experiments
        results = pipeline.run_experiments()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    # Quick test
    pipeline = SelfHealingPipeline()
    
    # Run single cycle with label noise
    print("\nRunning quick test with label noise...")
    results = pipeline.run_full_cycle(
        failure_mode='label_noise',
        failure_params={
            'condition': lambda x: x[0] > 0.5 and x[1] < 0,
            'noise_rate': 0.3
        },
        auto_heal=True
    )
    
    print("\n✓ Quick test complete!")
    print(f"✓ Final status: {results['status']}")