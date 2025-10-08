"""
Discovery History for storing and tracking algorithm evolution.
Simple storage with performance indexing and technique tracking.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime

N_RECENT_ALGORITHMS = 20  # default; can be overridden via config

class DiscoveryHistory:
    """Storage for algorithm discovery history and performance tracking."""

    def _get_ground_truth_metric(self, metrics: Dict[str, Any]) -> float:
        """Extract ground truth metric value with case-insensitive lookup."""
        # Try exact match first
        if self.performance_levels_ground_truth_label in metrics:
            return metrics[self.performance_levels_ground_truth_label]
        # Try case-insensitive match
        for key in metrics:
            if key.lower() == self.performance_levels_ground_truth_label.lower():
                return metrics[key]
        # Return default based on sense
        return 0 if self.performance_levels_ground_truth_label_sense == 'higher_is_better' else float('inf')
    
    def _classify_performance(self, metrics: Dict[str, Any]) -> str:
        """Classify algorithm performance based on metrics."""
        if not metrics:
            return 'unknown'
            
        # Check if evaluation was aborted
        if metrics.get('aborted', False):
            return 'incomplete'
            
        # For ground truth mode
        eval_mode = self.config.get('evaluation', {}).get('mode', 'ground_truth')
        if eval_mode == 'ground_truth':
            # Ensure the requested label is actually present before using fallback
            label = self.performance_levels_ground_truth_label
            present = any(k.lower() == label.lower() for k in metrics.keys())
            if not present:
                return 'unknown'
            ground_truth_value = self._get_ground_truth_metric(metrics)
            # Guard against NaN values
            try:
                if isinstance(ground_truth_value, float) and math.isnan(ground_truth_value):
                    return 'unknown'
            except Exception:
                pass
                
            if self.performance_levels_ground_truth_label_sense == 'higher_is_better':
                # Sort thresholds in descending order to check highest first
                sorted_levels = sorted(self.performance_levels_ground_truth.items(), 
                                     key=lambda x: x[1], reverse=True)
                for level, threshold in sorted_levels:
                    if ground_truth_value >= threshold:
                        return level
            else:
                # Sort thresholds in ascending order to check lowest first
                sorted_levels = sorted(self.performance_levels_ground_truth.items(), 
                                     key=lambda x: x[1])
                for level, threshold in sorted_levels:
                    if ground_truth_value <= threshold:
                        return level
            return 'poor'
                
            # For human/VLM modes, use structured evaluation
        else:
            qualitative_label = self.performance_levels_qualitative_label
            if 'structured_evaluation' in metrics and qualitative_label in metrics['structured_evaluation']:
                eval_data = metrics['structured_evaluation']
                qualitative_value = eval_data.get(qualitative_label, 0)
                # Guard against NaN values
                try:
                    if isinstance(qualitative_value, float) and math.isnan(qualitative_value):
                        return 'unknown'
                except Exception:
                    pass
                
                # Map quality score to performance levels
                if self.performance_levels_qualitative_label_sense == 'higher_is_better':
                    # Sort thresholds in descending order to check highest first
                    sorted_levels = sorted(self.performance_levels_qualitative.items(), 
                                         key=lambda x: x[1], reverse=True)
                    for level, threshold in sorted_levels:
                        if qualitative_value >= threshold:
                            return level
                else:
                    # Sort thresholds in ascending order to check lowest first
                    sorted_levels = sorted(self.performance_levels_qualitative.items(), 
                                         key=lambda x: x[1])
                    for level, threshold in sorted_levels:
                        if qualitative_value <= threshold:
                            return level
                return 'poor'
        return 'unknown'

    def _rebuild_indices(self) -> None:
        """Rebuild performance and technique indices."""
        self.performance_index.clear()
        
        for i, algo in enumerate(self.algorithms):
            # Performance index
            perf_level = self._classify_performance(algo.get('metrics', {}))
            self.performance_index[perf_level].append(i)
            
    
    def _load_existing(self) -> None:
        """Load existing history if available."""
        if not (self.save_path.exists() and self.save_path.is_dir()):
            return
            
        # Find the most recent history file
        history_files = sorted(self.save_path.glob(f"{self.name}_history_*.json"))
        if not history_files:
            return
            
        latest_file = history_files[-1]
        try:
            data = json.loads(latest_file.read_text())
            
            self.algorithms = data.get('algorithms', [])
            
            # Rebuild indices
            self._rebuild_indices()
            
        except Exception as e:
            logging.warning(f"Failed to load existing history file '{latest_file}': {e}. Starting fresh.")
    
    def __init__(self, config: Dict[str, Any], name: str):
        """Initialize discovery history with configuration."""
        self.config = config
        self.name = name
        # Use only generic 'history' section from config
        self.history_config = config.get('history', {})
        
        # Storage
        self.algorithms = []
        self.performance_index = defaultdict(list)  # Group by performance level
        
        # Limits from config
        self.compression_threshold = self.history_config.get('compression_threshold', 50)
        # Context sizing (configurable)
        self.recent_for_context = self.history_config.get('recent_for_context', N_RECENT_ALGORITHMS)
        self.suggestion_pool_last_k_high = self.history_config.get('suggestion_pool_last_k_high', 10)
        self.recent_suggestions_last_k = self.history_config.get('recent_suggestions_last_k', 5)
        
        # Compression policy configuration
        self.compression_moderate_share = self.history_config.get('compression_moderate_share', 0.25)  # Share of residual budget for moderate algorithms
        self.compression_min_moderate = self.history_config.get('compression_min_moderate', 5)  # Minimum number of moderate algorithms to keep
        self.compression_keep_top_n = self.history_config.get('compression_keep_top_n', 3)  # Always keep top N performers regardless of classification
        
        # Performance thresholds from config
        analysis_config = config.get('analysis', {})
        
        # Ground truth performance configuration
        # NOTE: When changing performance_levels_ground_truth_label, you MUST also update
        # performance_levels_ground_truth thresholds to match your metric's scale
        self.performance_levels_ground_truth_label = analysis_config.get('performance_levels_ground_truth_label', 'ssim')
        self.performance_levels_ground_truth_label_sense = analysis_config.get('performance_levels_ground_truth_label_sense', 'higher_is_better')
        self.performance_levels_ground_truth = {
            'excellent': 0.9,
            'good': 0.8,
            'moderate': 0.6,
            **analysis_config.get('performance_levels_ground_truth', {})
        }
                
        # Qualitative performance configuration
        self.performance_levels_qualitative_label = analysis_config.get('performance_levels_qualitative_label', 'quality_score')
        assert self.performance_levels_qualitative_label != 'feedback', "performance_levels_qualitative_label must be a numerical label"
        self.performance_levels_qualitative_label_sense = analysis_config.get('performance_levels_qualitative_label_sense', 'higher_is_better')
        self.performance_levels_qualitative = {
            'excellent': 0.9,
            'good': 0.7,
            'moderate': 0.5,
            **analysis_config.get('performance_levels_qualitative', {})
        }
        
        # Load existing history if available
        self.save_path = Path(self.config.get('save_path', f'./{self.name}_history/'))
        self._load_existing()

    def _compress(self) -> None:
        """Compress knowledge base by removing poor performers."""
        # Compressing knowledge base
        
        # Always keep the top N performers regardless of classification
        keep_indices = set()
        top_performers = self.get_top_performers(self.compression_keep_top_n)
        for top_algo in top_performers:
            # Find the index of this algorithm in our current list
            for i, algo in enumerate(self.algorithms):
                if algo.get('id') == top_algo.get('id'):
                    keep_indices.add(i)
                    break
        
        # Keep all excellent and good algorithms
        for level in ['excellent', 'good']:
            keep_indices.update(self.performance_index.get(level, []))
        
        # Keep some moderate algorithms - select best ones by metric value
        moderate_indices = self.performance_index.get('moderate', [])
        if moderate_indices:
            # Calculate residual budget after keeping excellent and good
            algorithms_kept = len(keep_indices)
            residual_budget = max(0, self.compression_threshold - algorithms_kept)
            # Keep based on configuration: share of residual budget and minimum count
            keep_count = min(len(moderate_indices), 
                           max(int(residual_budget * self.compression_moderate_share),
                               self.compression_min_moderate))
            # Guard against keep_count being 0
            if keep_count > 0:
                # Get moderate algorithms and sort by metric value
                moderate_algos = [(i, self.algorithms[i]) for i in moderate_indices]
                
                if self.config['evaluation']['mode'] == 'ground_truth':
                    # Sort by ground truth metric
                    def get_metric_value(item):
                        _, algo = item
                        return self._get_ground_truth_metric(algo.get('metrics', {}))
                    
                    sorted_moderates = sorted(moderate_algos, 
                                            key=get_metric_value,
                                            reverse=(self.performance_levels_ground_truth_label_sense == 'higher_is_better'))
                else:
                    # Sort by qualitative metric
                    sorted_moderates = sorted(moderate_algos,
                                            key=lambda item: item[1].get('metrics', {}).get('structured_evaluation', {}).get(self.performance_levels_qualitative_label, 0),
                                            reverse=(self.performance_levels_qualitative_label_sense == 'higher_is_better'))
                
                # Keep the best moderate algorithms
                best_moderate_indices = [item[0] for item in sorted_moderates[:keep_count]]
                keep_indices.update(best_moderate_indices)
        
        # If no excellent, good, or moderate algorithms found, keep best poor performers
        if not keep_indices:
            poor_indices = self.performance_index.get('poor', [])
            if poor_indices:
                # Calculate how many poor algorithms to keep
                keep_count = min(len(poor_indices), 
                               max(self.compression_min_moderate, 
                                   int(self.compression_threshold * self.compression_moderate_share)))
                
                # Get poor algorithms and sort by metric value
                poor_algos = [(i, self.algorithms[i]) for i in poor_indices]
                
                if self.config['evaluation']['mode'] == 'ground_truth':
                    # Sort by ground truth metric
                    def get_metric_value(item):
                        _, algo = item
                        return self._get_ground_truth_metric(algo.get('metrics', {}))
                    
                    sorted_poor = sorted(poor_algos, 
                                       key=get_metric_value,
                                       reverse=(self.performance_levels_ground_truth_label_sense == 'higher_is_better'))
                else:
                    # Sort by qualitative metric
                    sorted_poor = sorted(poor_algos,
                                       key=lambda item: item[1].get('metrics', {}).get('structured_evaluation', {}).get(self.performance_levels_qualitative_label, 0),
                                       reverse=(self.performance_levels_qualitative_label_sense == 'higher_is_better'))
                
                # Keep the best poor algorithms
                best_poor_indices = [item[0] for item in sorted_poor[:keep_count]]
                keep_indices.update(best_poor_indices)
        
        # Keep recent algorithms regardless of performance (but exclude incomplete)
        recent_count = min(self.recent_for_context, len(self.algorithms))
        if recent_count > 0:
            for i in range(len(self.algorithms) - recent_count, len(self.algorithms)):
                perf_level = self._classify_performance(self.algorithms[i].get('metrics', {}))
                if perf_level != 'incomplete':
                    keep_indices.add(i)
        
        # Filter algorithms
        keep_indices = sorted(list(keep_indices))
        self.algorithms = [self.algorithms[i] for i in keep_indices]
        
        # Rebuild indices (this will clear old indices and rebuild from new algorithm positions)
        self._rebuild_indices()
        
        # Compression complete
 
       
    def _summarize_algorithm(self, algorithm: Dict[str, Any], keep_code: bool = False) -> Dict[str, Any]:
        """Summarize a single algorithm for context."""
        if not algorithm:
            return {}
        
        # Extract only essential information
        summary = {
            'id': algorithm.get('id', 'unknown'),
            'techniques': algorithm.get('analysis', {}).get('techniques', []),
            'suggestions': algorithm.get('analysis', {}).get('suggestions', []),
            'action': algorithm.get('action', 'unknown'),
            'metrics': algorithm.get('metrics', {})
        }

        if keep_code:
            summary['code'] = algorithm.get('code', '')

        return summary
    
    def summarize_algorithms(self, algorithms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize algorithms for context to prevent token bloat."""
        if not algorithms:
            return []
        
        return [self._summarize_algorithm(algo) for algo in algorithms]
   
    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get N most recent algorithms."""
        n = self.recent_for_context if n is None else n
        return self.algorithms[-n:] if self.algorithms else []

    def get_best_algorithm(self) -> Optional[Dict[str, Any]]:
        """Get the best performing algorithm."""
        top_performers = self.get_top_performers(1)
        return top_performers[0] if top_performers else None
        
    
    def get_top_performers(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N performing algorithms."""
        if not self.algorithms:
            return []
        # Consider only successful algorithms
        candidates = [a for a in self.algorithms if a.get('success', False)]
        if not candidates:
            return []

        if self.config['evaluation']['mode'] == 'ground_truth':
            # Use helper for ground truth metric extraction with NaN handling
            def get_metric_value_safe(algo):
                val = self._get_ground_truth_metric(algo.get('metrics', {}))
                # Handle NaN by returning worst possible value; guard non-floats
                try:
                    if math.isnan(val):
                        return float('-inf') if self.performance_levels_ground_truth_label_sense == 'higher_is_better' else float('inf')
                except TypeError:
                    pass
                return val
            
            return sorted(candidates, 
                       key=get_metric_value_safe,
                       reverse=(self.performance_levels_ground_truth_label_sense == 'higher_is_better'))[:n]
        else:
            # Filter out algorithms without the qualitative metric
            algorithms_with_metric = [
                algo for algo in candidates 
                if self.performance_levels_qualitative_label in algo.get('metrics', {}).get('structured_evaluation', {})
            ]
            
            if not algorithms_with_metric:
                if self.algorithms:
                    # Warning: No algorithms found with qualitative metric
                    pass
                return []
                
            return sorted(algorithms_with_metric, 
                       key=lambda algo: algo['metrics']['structured_evaluation'].get(self.performance_levels_qualitative_label, 0),
                       reverse=(self.performance_levels_qualitative_label_sense == 'higher_is_better'))[:n]
    

    def get_by_performance(self, level: str) -> List[Dict[str, Any]]:
        """Get algorithms by performance level."""
        indices = self.performance_index.get(level, [])
        return [self.algorithms[i] for i in indices if i < len(self.algorithms)]
    
   
    def get_performance_distribution(self) -> Dict[str, int]:
        """Get distribution of algorithms by performance level."""
        distribution = {}
        for level in ['excellent', 'good', 'moderate', 'poor', 'unknown', 'incomplete']:
            distribution[level] = len(self.performance_index.get(level, []))
        return distribution
    

    def get_context(self) -> Dict[str, Any]:
        """Get context for LLM generation."""
        context = {
            'recent_algorithms': self.summarize_algorithms(self.get_recent(self.recent_for_context)),
            'best_performance': self._summarize_algorithm(self.get_best_algorithm(), keep_code=True) if self.get_best_algorithm() else {},
        }
        
        # Aggregate suggestions from high-performing algorithms
        good_and_excellent = []
        for level in ['excellent', 'good']:
            good_and_excellent.extend(self.get_by_performance(level))
        
        # Collect unique suggestions prioritizing recent and high-performing algorithms
        suggestions_dict = {}
        for algo in good_and_excellent[-self.suggestion_pool_last_k_high:]:
            for suggestion in algo.get('analysis', {}).get('suggestions', []):
                if suggestion and suggestion not in suggestions_dict:
                    suggestions_dict[suggestion] = {
                        'text': suggestion,
                        'from_algorithm': algo.get('id'),
                        'performance': self._classify_performance(algo.get('metrics', {}))
                    }
        
        # Also add suggestions from recent algorithms if they had moderate performance
        for algo in self.get_recent(self.recent_suggestions_last_k):
            perf_level = self._classify_performance(algo.get('metrics', {}))
            if perf_level in ['moderate', 'good', 'excellent']:
                for suggestion in algo.get('analysis', {}).get('suggestions', []):
                    if suggestion and suggestion not in suggestions_dict:
                        suggestions_dict[suggestion] = {
                            'text': suggestion,
                            'from_algorithm': algo.get('id'),
                            'performance': perf_level
                        }
        
        context['aggregated_suggestions'] = list(suggestions_dict.values())
        
        return context
    
    def add(self, algorithm: Dict[str, Any]) -> None:
        """Add algorithm to discovery history."""
        # Add timestamp
        algorithm['timestamp'] = datetime.now().isoformat()
        
        # Normalize action names for consistency
        if 'action' in algorithm:
            action_map = {
                'generate': 'generated',
                'tune': 'tuned',
                'evolve': 'evolved'
            }
            algorithm['action'] = action_map.get(algorithm['action'], algorithm['action'])
        
        # Add to main storage
        self.algorithms.append(algorithm)
        
        # Index by performance
        perf_level = self._classify_performance(algorithm.get('metrics', {}))
        self.performance_index[perf_level].append(len(self.algorithms) - 1)
        
        # Check if compression needed
        if len(self.algorithms) >= self.compression_threshold:
            self._compress()
            self.save()
            
        # Algorithm added to knowledge base

    
    def save(self, path: Optional[Path] = None) -> None:
        """Save current history to file."""
        save_path = path or self.save_path / f"{self.name}_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_path.parent.mkdir(exist_ok=True)
        
        data = {
            'algorithms': self.algorithms,
            'performance_index': dict(self.performance_index),
            'metadata': {
                'total_algorithms': len(self.algorithms),
                'save_time': datetime.now().isoformat()
            }
        }
        
        # sanitize Inf/NaN by forcing strict JSON and catching failures
        try:
            save_path.write_text(json.dumps(data, indent=2, allow_nan=False))
        except (TypeError, ValueError):
            # Fallback: sanitize numerics to strings
            from math import isinf, isnan
            def _sf(o):
                if isinstance(o, float) and (isinf(o) or isnan(o)):
                    return "Infinity" if isinf(o) else "NaN"
                return o
            def _walk(x):
                if isinstance(x, dict):
                    return {k: _walk(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [_walk(v) for v in x]
                return _sf(x)
            save_path.write_text(json.dumps(_walk(data), indent=2, allow_nan=False))
            
        # History saved
    
    def size(self) -> int:
        """Get current size of knowledge base."""
        return len(self.algorithms)
    
    def export(self) -> Dict[str, Any]:
        """Export history data for session."""
        # Calculate technique frequency
        technique_freq = {}
        for algo in self.algorithms:
            techniques = algo.get('analysis', {}).get('techniques', [])
            for tech in techniques:
                technique_freq[tech] = technique_freq.get(tech, 0) + 1
        
        # Calculate evolution stats
        evolution_stats = {
            'mutations': sum(1 for a in self.algorithms if a.get('action') == 'evolved' and a.get('strategy') == 'mutation'),
            'crossovers': sum(1 for a in self.algorithms if a.get('action') == 'evolved' and a.get('strategy') == 'crossover'),
            'generated': sum(1 for a in self.algorithms if a.get('action') == 'generated'),
            'tuned': sum(1 for a in self.algorithms if a.get('action') == 'tuned'),
            'evolved': sum(1 for a in self.algorithms if a.get('action') == 'evolved')  # Total evolved
        }
        
        return {
            'algorithms': self.summarize_algorithms(self.algorithms),
            'total_algorithms': len(self.algorithms),
            'performance_distribution': self.get_performance_distribution(),
            'technique_frequency': technique_freq,
            'evolution_stats': evolution_stats,
            'recent_algorithms': self.summarize_algorithms(self.get_recent()),
            'top_performers': self.summarize_algorithms(self.get_top_performers()),
            'metadata': {
                'name': self.name,
                'compression_threshold': self.compression_threshold
            }
        }
    
