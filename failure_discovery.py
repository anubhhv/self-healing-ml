"""
Module 1: Failure Discovery Engine
Finds systematic error patterns, clusters failures, detects blind spots.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FailureCluster:
    """Represents a discovered failure pattern."""
    cluster_id: int
    size: int
    percentage: float
    indices: np.ndarray
    feature_pattern: Dict
    statistical_significance: float
    avg_confidence: float
    samples: np.ndarray


class ErrorCollector:
    """Collects predictions and tracks failures."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.predictions = []
        self.confidences = []
        self.true_labels = []
        self.features = []
        self.activations = []
        
    def collect(self, X, y, return_activations=True):
        """
        Run inference and collect all data.
        
        Returns:
            failures: Boolean array of failed predictions
            metadata: Dict with predictions, confidences, etc.
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.model(X_tensor)
            
            # Get predictions and confidences
            probs = torch.softmax(output, dim=1)
            confidence, predictions = torch.max(probs, dim=1)
            
            predictions = predictions.cpu().numpy()
            confidence = confidence.cpu().numpy()
            
            # Get intermediate activations (for clustering)
            if return_activations and hasattr(self.model, 'network'):
                # For MLP, get second-to-last layer
                activations = X_tensor
                for layer in list(self.model.network.children())[:-1]:
                    activations = layer(activations)
                activations = activations.cpu().numpy()
            else:
                activations = X  # Fall back to input features
        
        # Identify failures
        failures = predictions != y
        
        metadata = {
            'predictions': predictions,
            'confidences': confidence,
            'true_labels': y,
            'features': X,
            'activations': activations,
            'failures': failures,
            'num_failures': failures.sum(),
            'failure_rate': failures.mean()
        }
        
        return failures, metadata


class PatternMiner:
    """Discovers interpretable failure patterns using subgroup discovery."""
    
    def __init__(self, max_depth=3, min_samples_leaf=20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
    def find_patterns(self, X, failures, feature_names=None):
        """
        Find feature combinations that predict failures.
        
        Returns:
            List of patterns (rules) that lead to failures
        """
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(X.shape[1])]
        
        # Train decision tree to predict failures
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        tree.fit(X, failures.astype(int))
        
        # Extract rules from tree
        patterns = self._extract_rules(tree, feature_names, X, failures)
        
        return patterns
    
    def _extract_rules(self, tree, feature_names, X, failures):
        """Extract interpretable rules from decision tree."""
        from sklearn.tree import _tree
        
        patterns = []
        
        def recurse(node, conditions):
            if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]
                
                # Left child (feature <= threshold)
                left_conditions = conditions + [(feature_names[feature], '<=', threshold)]
                recurse(tree.tree_.children_left[node], left_conditions)
                
                # Right child (feature > threshold)
                right_conditions = conditions + [(feature_names[feature], '>', threshold)]
                recurse(tree.tree_.children_right[node], right_conditions)
            else:
                # Leaf node
                samples_in_node = tree.tree_.n_node_samples[node]
                failures_in_node = tree.tree_.value[node][0][1]  # Class 1 (failure)
                
                if failures_in_node > 0 and samples_in_node >= self.min_samples_leaf:
                    # Check which samples satisfy this rule
                    mask = np.ones(len(X), dtype=bool)
                    for feat_name, op, val in conditions:
                        feat_idx = feature_names.index(feat_name)
                        if op == '<=':
                            mask &= X[:, feat_idx] <= val
                        else:
                            mask &= X[:, feat_idx] > val
                    
                    if mask.sum() > 0:
                        pattern = {
                            'conditions': conditions,
                            'num_samples': int(mask.sum()),
                            'num_failures': int(failures[mask].sum()),
                            'failure_rate': float(failures[mask].mean()),
                            'indices': np.where(mask)[0]
                        }
                        patterns.append(pattern)
        
        recurse(0, [])
        
        # Sort by failure rate
        patterns.sort(key=lambda x: x['failure_rate'], reverse=True)
        
        return patterns


class FailureClusterer:
    """Clusters failures in representation space."""
    
    def __init__(self, min_cluster_size=10, min_samples=5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
    def cluster(self, activations, failures, X, confidences):
        """
        Cluster failed samples in representation space.
        
        Returns:
            List of FailureCluster objects
        """
        if failures.sum() < self.min_cluster_size:
            print("Too few failures to cluster")
            return []
        
        # Get failed samples
        failed_activations = activations[failures]
        failed_features = X[failures]
        failed_confidences = confidences[failures]
        failed_indices = np.where(failures)[0]
        
        # Cluster using HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(failed_activations)
        
        # Create FailureCluster objects
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            mask = labels == label
            cluster_indices = failed_indices[mask]
            
            # Calculate feature statistics for this cluster
            cluster_features = failed_features[mask]
            feature_pattern = self._analyze_feature_pattern(cluster_features, X)
            
            cluster = FailureCluster(
                cluster_id=int(label),
                size=int(mask.sum()),
                percentage=float(mask.sum() / failures.sum() * 100),
                indices=cluster_indices,
                feature_pattern=feature_pattern,
                statistical_significance=self._compute_significance(cluster_features, X),
                avg_confidence=float(failed_confidences[mask].mean()),
                samples=cluster_features
            )
            clusters.append(cluster)
        
        # Sort by size
        clusters.sort(key=lambda x: x.size, reverse=True)
        
        return clusters
    
    def _analyze_feature_pattern(self, cluster_features, all_features):
        """Analyze what makes this cluster unique."""
        patterns = {}
        
        for i in range(cluster_features.shape[1]):
            cluster_mean = cluster_features[:, i].mean()
            cluster_std = cluster_features[:, i].std()
            overall_mean = all_features[:, i].mean()
            overall_std = all_features[:, i].std()
            
            # Check if this feature is significantly different
            if abs(cluster_mean - overall_mean) > 0.5 * overall_std:
                patterns[f'feature_{i}'] = {
                    'cluster_mean': float(cluster_mean),
                    'overall_mean': float(overall_mean),
                    'difference': float(cluster_mean - overall_mean),
                    'cluster_std': float(cluster_std)
                }
        
        return patterns
    
    def _compute_significance(self, cluster_features, all_features):
        """Compute statistical significance of this cluster."""
        # Use KS test on first principal feature
        if len(cluster_features) < 5:
            return 0.0
        
        statistic, pvalue = ks_2samp(
            cluster_features[:, 0],
            all_features[:, 0]
        )
        
        return float(1 - pvalue)  # Higher = more significant


class BlindSpotDetector:
    """Detects regions with high errors and low training density."""
    
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        
    def detect(self, X_train, X_test, failures, feature_names=None):
        """
        Find blind spots: regions with high test errors but low training coverage.
        
        Returns:
            List of blind spot regions
        """
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        blind_spots = []
        
        # For each feature dimension
        for feat_idx in range(X_train.shape[1]):
            train_feat = X_train[:, feat_idx]
            test_feat = X_test[:, feat_idx]
            
            # Discretize into bins
            bins = np.linspace(
                min(train_feat.min(), test_feat.min()),
                max(train_feat.max(), test_feat.max()),
                self.n_bins + 1
            )
            
            train_hist, _ = np.histogram(train_feat, bins=bins)
            test_hist, _ = np.histogram(test_feat, bins=bins)
            
            # Count failures in each bin
            failure_hist = np.zeros(self.n_bins)
            for i in range(self.n_bins):
                in_bin = (test_feat >= bins[i]) & (test_feat < bins[i+1])
                if in_bin.sum() > 0:
                    failure_hist[i] = failures[in_bin].sum()
            
            # Find bins with: low training density + high failure rate
            train_density = train_hist / (train_hist.sum() + 1e-6)
            test_density = test_hist / (test_hist.sum() + 1e-6)
            failure_rate = failure_hist / (test_hist + 1e-6)
            
            for i in range(self.n_bins):
                if train_density[i] < 0.05 and failure_rate[i] > 0.3:
                    blind_spots.append({
                        'feature': feature_names[feat_idx],
                        'feature_idx': feat_idx,
                        'range': (float(bins[i]), float(bins[i+1])),
                        'train_density': float(train_density[i]),
                        'test_density': float(test_density[i]),
                        'failure_rate': float(failure_rate[i]),
                        'num_failures': int(failure_hist[i]),
                        'severity': float(failure_rate[i] * test_density[i])
                    })
        
        # Sort by severity
        blind_spots.sort(key=lambda x: x['severity'], reverse=True)
        
        return blind_spots


class FailureDiscoveryEngine:
    """Main engine that orchestrates all failure discovery components."""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f'f{i}' for i in range(X_test.shape[1])]
        
        self.collector = ErrorCollector(model)
        self.pattern_miner = PatternMiner()
        self.clusterer = FailureClusterer()
        self.blind_spot_detector = BlindSpotDetector()
        
    def discover(self):
        """
        Run full failure discovery pipeline.
        
        Returns:
            Dict with all discovered failure information
        """
        print("\n" + "="*70)
        print("FAILURE DISCOVERY ENGINE")
        print("="*70)
        
        # Step 1: Collect errors
        print("\n[1/4] Collecting errors...")
        failures, metadata = self.collector.collect(self.X_test, self.y_test)
        print(f"  ✓ Found {metadata['num_failures']} failures ({metadata['failure_rate']*100:.1f}%)")
        
        # Step 2: Mine patterns
        print("\n[2/4] Mining failure patterns...")
        patterns = self.pattern_miner.find_patterns(
            self.X_test, failures, self.feature_names
        )
        print(f"  ✓ Discovered {len(patterns)} interpretable patterns")
        for i, p in enumerate(patterns[:3]):  # Show top 3
            print(f"    Pattern {i+1}: {p['num_failures']}/{p['num_samples']} failures "
                  f"({p['failure_rate']*100:.1f}%)")
            for cond in p['conditions'][:2]:  # Show first 2 conditions
                print(f"      - {cond[0]} {cond[1]} {cond[2]:.3f}")
        
        # Step 3: Cluster failures
        print("\n[3/4] Clustering failures...")
        clusters = self.clusterer.cluster(
            metadata['activations'],
            failures,
            self.X_test,
            metadata['confidences']
        )
        print(f"  ✓ Found {len(clusters)} failure clusters")
        for i, c in enumerate(clusters):
            print(f"    Cluster {i+1}: {c.size} samples ({c.percentage:.1f}%), "
                  f"significance={c.statistical_significance:.3f}")
        
        # Step 4: Detect blind spots
        print("\n[4/4] Detecting blind spots...")
        blind_spots = self.blind_spot_detector.detect(
            self.X_train, self.X_test, failures, self.feature_names
        )
        print(f"  ✓ Found {len(blind_spots)} blind spots")
        for i, bs in enumerate(blind_spots[:3]):  # Show top 3
            print(f"    Blind spot {i+1}: {bs['feature']} in "
                  f"[{bs['range'][0]:.2f}, {bs['range'][1]:.2f}], "
                  f"failure_rate={bs['failure_rate']*100:.1f}%")
        
        print("\n" + "="*70)
        
        return {
            'metadata': metadata,
            'patterns': patterns,
            'clusters': clusters,
            'blind_spots': blind_spots,
            'summary': {
                'total_failures': metadata['num_failures'],
                'failure_rate': metadata['failure_rate'],
                'num_patterns': len(patterns),
                'num_clusters': len(clusters),
                'num_blind_spots': len(blind_spots)
            }
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def load_model_and_data(checkpoint_path, data_path):
    """Load trained model and data."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Reconstruct model
    from train import MLP
    model = MLP(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return model, data, checkpoint


if __name__ == "__main__":
    
    print("\nFailure Discovery - Testing on Trained Models")
    print("="*70)
    
    # Test on each failure mode
    failure_modes = ['label_noise', 'missing_feature', 'spurious', 'data_shift']
    
    results = {}
    
    for failure_mode in failure_modes:
        print(f"\n\n{'='*70}")
        print(f"Testing: {failure_mode.upper()}")
        print('='*70)
        
        try:
            # Load model and data
            checkpoint_path = f'./checkpoints/model_{failure_mode}.pt'
            data_path = f'./checkpoints/data_{failure_mode}.pkl'
            
            model, data, checkpoint = load_model_and_data(checkpoint_path, data_path)
            
            # Run discovery
            engine = FailureDiscoveryEngine(
                model=model,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_test=data['X_test'],
                y_test=data['y_test']
            )
            
            discovery_results = engine.discover()
            results[failure_mode] = discovery_results
            
        except Exception as e:
            print(f"Error processing {failure_mode}: {e}")
    
    # Save results
    output_path = './checkpoints/discovery_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n\n{'='*70}")
    print("✓ Failure discovery complete!")
    print(f"✓ Results saved to {output_path}")
    print('='*70)