"""
Algorithm Discovery Framework.
Main entry point for discovering novel ptychographic reconstruction algorithms.
"""

import json
import time
import uuid
import signal
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from .llm_engine import LLMEngine
from .recon_evaluator import ReconEvaluator
from .history import DiscoveryHistory
from .checkpoint import CheckpointManager
from .exceptions import ConfigurationError, PromptError, GenerationError
from .logging import get_logger

REQUIRED_PROMPTS = ['algorithm_correction', 
                    'analysis', 
                    'web_search', 
                    'discovery', 
                    'evolution_crossover', 
                    'evolution_mutation', 
                    'parameter_tuning']


class AlgorithmDiscovery:
    """Main class for algorithm discovery using LLM and evaluation."""
    
    def __init__(self, config: Dict[str, Any], experiment_context: Optional[str] = None, example_algorithms: Optional[List[str]] = None, verbose: bool = False, debug: bool = False):
        """
        Initialize the discovery framework.
        
        Args:
            config: Configuration dictionary
            experiment_context: Optional experiment description for initial research
            example_algorithms: Optional list of example algorithms for initial research
            verbose: Enable verbose logging for debugging
            debug: Enable debug logging with input/output details
        """
        self.config = self._validate_config(config)
        self.name = self.config.get('name', 'discovery')
        self.max_attempts = self.config.get('max_attempts', 100)
        self.max_correction_attempts = self.config.get('max_correction_attempts', 2)
        self.n_warmup_iterations = self.config.get('n_warmup_iterations', 5)
        self.n_target_algorithms = self.config.get('n_target_algorithms', 10)
        
        # Verbose mode - can be set via parameter or config
        self.verbose = verbose or self.config.get('verbose', False)
        # Debug mode - can be set via parameter or config
        self.debug = debug or self.config.get('debug', False)
        
        # Initialize unified logger
        self.log = get_logger(__name__, verbose=self.verbose, debug=self.debug)
        
        self.log.init(f"Initializing AlgorithmDiscovery '{self.name}'")
        self.log.init(f"Max attempts: {self.max_attempts}")
        self.log.init(f"Target algorithms: {self.n_target_algorithms}")
        self.log.init(f"Evaluation mode: {self.config.get('evaluation', {}).get('mode', 'unknown')}")
        
        if self.debug:
            self.log.debug("Debug mode enabled - will show input/output for all calls")
        
        self.llm = LLMEngine(self.config, verbose=self.verbose, debug=self.debug)
        self.evaluator = ReconEvaluator(self.config, self.llm, verbose=self.verbose, debug=self.debug)
        self.history = DiscoveryHistory(self.config, self.name)
        
        # Initialize checkpoint manager
        save_path = Path(self.config.get('save_path', './results'))
        checkpoint_interval = self.config.get('checkpoint_interval', 10)
        self.checkpoint_manager = CheckpointManager(save_path, self.name, checkpoint_interval)
        
        self.log.init("Components initialized successfully")
        self.log.init(f"History contains {self.history.size()} existing algorithms")
        self.log.init(f"Checkpoint interval: every {checkpoint_interval} algorithms")
        
        # Experiment context
        self.experiment_context = experiment_context
        self.example_algorithms = example_algorithms
        # Statistics tracking
        self.stats = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'tuning_attempts': 0,
            'start_time': time.time()
        }

        # --- Action selection & early stop policies (configurable) ---
        dsp = self.config.get('discovery', {}).get('action_policy', {})
        self.action_policy = {
            'honor_suggestion_min_level': dsp.get('honor_suggestion_min_level', 'moderate'),
            'warmup_iterations': dsp.get('warmup_iterations', self.n_warmup_iterations),
            'generate_batch_size': int(dsp.get('generate_batch_size', 1)),
            'tune_min_excellent': dsp.get('tune', {}).get('min_excellent', 3),
            'tune_every_k': dsp.get('tune', {}).get('every_k', 3),
            'evolve_min_successful': dsp.get('evolve', {}).get('min_successful', 5),
            'evolve_every_k': dsp.get('evolve', {}).get('every_k', 2),
            'fallback_action': dsp.get('fallback_action', 'generate'),
        }

        esp = self.config.get('discovery', {}).get('early_stop', {})
        self.early_stop = {
            'target_excellent': esp.get('target_excellent', self.n_target_algorithms),
            'target_good_plus_excellent': esp.get('target_good_plus_excellent', self.n_target_algorithms),
            'target_any_successful': esp.get('target_any_successful', self.n_target_algorithms),
            'require_min_good_plus_excellent': esp.get('require_min_good_plus_excellent', 5),
            'require_at_least_one_excellent': bool(esp.get('require_at_least_one_excellent', True)),
        }

        # Error policy (max consecutive errors before abort)
        self.max_consecutive_errors = int(self.config.get('discovery', {}).get('max_consecutive_errors', 5))

        # Log resolved policies
        self.log.init(f"Action policy: {self.action_policy}")
        self.log.init(f"Early-stop: {self.early_stop}")
        
        # Load checkpoint if resuming
        self._load_checkpoint()
        
        # Set up signal handlers for graceful shutdown
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        except (AttributeError, ValueError):
            self._original_sigterm = None
        self._shutdown_requested = False

    @staticmethod
    def _perf_rank(level: str) -> int:
        # Higher is better; unknown/incomplete are lowest
        order = {'unknown': 0, 'incomplete': 0, 'poor': 1, 'moderate': 2, 'good': 3, 'excellent': 4}
        return order.get(level, 0)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
        self.log.info(f"[SIGNAL] Received {signal_name} - saving checkpoint...")
        
        # Save checkpoint immediately
        try:
            self._save_checkpoint(force=True)
            self.log.discovery(f"[DISCOVERY] Checkpoint saved successfully!")
            self.log.discovery(f"[DISCOVERY] To resume, run with the same config and name: '{self.name}'")
        except Exception as e:
            self.log.error(f"[ERROR] Failed to save checkpoint on signal: {e}")
        
        # For SIGTERM, set flag and let main loop exit gracefully
        if signum == signal.SIGTERM:
            self._shutdown_requested = True
        else:
            # For SIGINT (Ctrl+C), restore original handler and re-raise
            signal.signal(signal.SIGINT, self._original_sigint)
            raise KeyboardInterrupt()
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and prepare configuration."""
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Load all required prompt templates from package prompts folder
        prompts_dir = Path(__file__).parent / 'prompts'
        # Preserve existing prompts from config, if any - handle None value explicitly
        if 'prompts' not in config or config['prompts'] is None:
            config['prompts'] = {}
            
        missing_prompts = []
        
        for prompt_name in REQUIRED_PROMPTS:
            # If already in config, use that
            if prompt_name in config['prompts']:
                continue
            
            # Look for template file
            prompt_file = prompts_dir / f"{prompt_name}.md"
            if prompt_file.exists():
                # Use absolute path
                full_path = prompt_file.resolve()
                config['prompts'][prompt_name] = full_path.read_text()
            else:
                missing_prompts.append(prompt_name)
        
        if missing_prompts:
            error_msg = f"Missing required prompt templates: {', '.join(missing_prompts)}"
            error_msg += f"Expected to find them in: {prompts_dir}"
            error_msg += "Ensure all prompt template files are present."
            raise PromptError(error_msg)
        
        # Ensure required top-level keys exist
        required_keys = ['llm', 'reconstruction']
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Missing required configuration key: {key}")
        
        # Set defaults
        config.setdefault('save_path', './results')
        config.setdefault('max_attempts', 100)
        config.setdefault('max_correction_attempts', 2)
        config.setdefault('n_warmup_iterations', 5)
        config.setdefault('n_target_algorithms', 10)
        config.setdefault('checkpoint_interval', 10)
        
        # Evaluation defaults
        if 'evaluation' not in config:
            config['evaluation'] = {}
        config['evaluation'].setdefault('mode', 'ground_truth')
        config['evaluation'].setdefault('iterations', 500)
        
        return config
    
    def gather_experiment_context(self) -> str:
        """Interactively gather experiment context from the user."""
        if self.experiment_context:
            return self.experiment_context
            
        # Skip interaction if running in automated mode
        if self.config.get('automated', False):
            return "Automated mode - no experiment context provided"
            
        # Interactive prompt
        self.log.discovery("EXPERIMENT CONTEXT GATHERING")
        self.log.discovery("Please provide information about your ptychography experiment.")
        self.log.discovery("Include details such as:")
        self.log.discovery("- Sample type and properties")
        self.log.discovery("- Experimental setup (beam energy, detector, etc.)")
        self.log.discovery("- Expected challenges or artifacts")
        self.log.discovery("- Any specific requirements or constraints")
        self.log.prompt("Enter your description (press Enter on an empty line when done):\n")
        
        lines = []
        while True:
            try:
                line = input()
                if not line:
                    break
                lines.append(line)
            except EOFError:
                break
                
        context = '\n'.join(lines)
        if not context:
            context = "No specific experiment context provided"
            
        self.log.discovery("Thank you! Context recorded.")

        return context
    
    def _decide_next_action(self) -> str:
        """Decide the next action based on history and performance."""
        # Check if previous algorithm had a suggestion AND was performing well
        if self.history.size() > 0:
            recent_algos = self.history.get_recent(1)
            if recent_algos:
                last_algo = recent_algos[0]
                if 'suggested_action' in last_algo.get('metrics', {}):
                    # Only honor suggestion if the algorithm performed at least moderately
                    perf_level = self.history._classify_performance(last_algo.get('metrics', {}))
                    min_level = self.action_policy['honor_suggestion_min_level']
                    if self._perf_rank(perf_level) >= self._perf_rank(min_level):
                        action = last_algo['metrics']['suggested_action']
                        self.log.decision(f"Using evaluator suggestion from {perf_level} algorithm (>= {min_level}): {action}")
                        return action
                    self.log.decision(f"Ignoring suggestion from {perf_level} (< {min_level})")
                
        # Simple heuristic based on algorithm count and performance
        excellent_algos = self.history.get_by_performance('excellent')
        good_algos = self.history.get_by_performance('good')
        moderate_algos = self.history.get_by_performance('moderate')
        
        num_excellent = len(excellent_algos)
        num_good = len(good_algos)
        num_moderate = len(moderate_algos)
        total_successful = num_excellent + num_good + num_moderate
        
        # Early phase: focus on generation
        if self.stats['total_generated'] < self.action_policy['warmup_iterations']:
            action = 'generate'
        # If we have enough excellent algorithms, tune them
        elif num_excellent >= self.action_policy['tune_min_excellent'] and \
             self.stats['total_generated'] % max(1, self.action_policy['tune_every_k']) == 0:
            action = 'tune'
        # If we have a good population, evolve
        elif total_successful >= self.action_policy['evolve_min_successful'] and \
             self.stats['total_generated'] % max(1, self.action_policy['evolve_every_k']) == 0:
            action = 'evolve'
        # Otherwise generate new
        else:
            action = self.action_policy.get('fallback_action', 'generate')
            
        self.log.decision(f"Action: {action} (excellent: {num_excellent}, good: {num_good}, moderate: {num_moderate})")
        
        return action
    
    def _generate_algorithm(self) -> Dict[str, Any]:
        """Generate a new algorithm using LLM."""
        self.log.generate("Starting new algorithm generation")
        
        # Build context by accumulating all available information
        context = self.experiment_context or ""
        
        # Add web search results if available
        if hasattr(self, 'web_search_results') and self.web_search_results:
            context += "\n\n" + "Common techniques: " + self.web_search_results
            
        # Add example algorithms if available
        if self.example_algorithms:
            context += "\n\n" + "Example algorithms: " + json.dumps(self.example_algorithms, indent=2)
        
        if self.history.size() == 0:
            history_context = {
                'experiment_context': context, 
                'recent_algorithms': [],
                'best_performance': {},
                'aggregated_suggestions': []
            }
        else:
            history_context = self.history.get_context()
            # Include the full context (experiment + web search + examples) for all algorithms
            history_context['experiment_context'] = context
        
        try:
            algorithm = self.llm.generate_algorithm(
                prompt=self.config['prompts']['discovery'],
                context=history_context,
            )
            # Ensure algorithm has an ID
            if 'id' not in algorithm:
                algorithm['id'] = uuid.uuid4().hex
            # Don't increment total_generated here - let the caller handle it
            algorithm['action'] = 'generate'
            
            return algorithm
            
        except Exception as e:
            self.log.error(f"Failed to generate algorithm: {e}")
            raise GenerationError(f"Algorithm generation failed: {str(e)}")
    
    def _tune_algorithms(self) -> List[Dict[str, Any]]:
        """Tune existing successful algorithms."""
        self.stats['tuning_attempts'] += 1
        
        self.log.tune(f"Starting parameter tuning (attempt #{self.stats['tuning_attempts']})")
        
        # Select algorithms for tuning
        candidates = self.history.get_top_performers()
        
        if not candidates:
            self.log.warning("No candidates for parameter tuning")
            return []
            
        self.log.tune(f"Found {len(candidates)} candidates for tuning")
        
        tuned = []
        for i, candidate in enumerate(candidates):
            self.log.tune(f"Tuning candidate {i+1}/{len(candidates)}: {candidate['id']}")
            
            # Tune using LLM
            tuned_algorithm = self.llm.tune_parameters(
                prompt=self.config['prompts']['parameter_tuning'],
                algorithm=candidate
            )
            if tuned_algorithm:
                # Ensure algorithm has an ID
                if 'id' not in tuned_algorithm:
                    tuned_algorithm['id'] = uuid.uuid4().hex
                # Don't increment total_generated here - let the caller handle it
                tuned_algorithm['action'] = 'tune'
                tuned_algorithm['parent_id'] = candidate['id']
                tuned.append(tuned_algorithm)
        
        return tuned
    
    def _evolve_algorithms(self) -> List[Dict[str, Any]]:
        """Evolve existing successful algorithms."""
        self.log.evolve("Starting algorithm evolution")
        
        population = self.history.get_top_performers()
        if not population:
            self.log.warning("No algorithms available for evolution")
            return []
        
        # Choose evolution strategy based on population
        strategy = 'crossover' if len(population) >= 2 else 'mutation'
        
        self.log.evolve(f"Using {strategy} strategy with {len(population)} algorithms")
        
        evolved = self.llm.evolve_algorithms(
            prompt=self.config['prompts'][f'evolution_{strategy}'],
            population=population,
            strategy=strategy
        )
        
        # Update metadata
        for algo in evolved:
            # Ensure algorithm has an ID
            if 'id' not in algo:
                algo['id'] = uuid.uuid4().hex
            # Don't increment total_generated here - let the caller handle it
            algo['action'] = 'evolve'
            # Use actual strategy if available (for odd population handling)
            if 'actual_strategy' in algo:
                algo['strategy'] = algo.pop('actual_strategy')
            else:
                algo['strategy'] = strategy
            # Parent info should already be set by llm_engine methods
            
        return evolved
    
    def _correct_algorithm(self, algorithm: Dict[str, Any], error: str) -> Optional[Dict[str, Any]]:
        """Attempt to correct an algorithm that failed evaluation."""
        max_attempts = self.max_correction_attempts
        
        for attempt in range(max_attempts):
            # Try to correct the error
            if error:
                self.log.correct(f"Attempting correction {attempt + 1}/{self.max_correction_attempts}")
                
                try:
                    corrected = self.llm.correct_algorithm(
                        prompt=self.config['prompts']['algorithm_correction'],
                        algorithm=algorithm,
                        error=error
                    )
                    if corrected:
                        # Ensure corrected algorithm has an ID
                        if 'id' not in corrected:
                            corrected['id'] = uuid.uuid4().hex
                        # Preserve metadata
                        corrected['generation'] = algorithm['generation']
                        corrected['action'] = algorithm.get('action', 'unknown')
                        corrected['correction_attempt'] = attempt + 1
                        corrected['original_id'] = algorithm.get('original_id', algorithm['id'])
                        
                        # Try evaluating the corrected version
                        success, metrics, error = self.evaluator.evaluate(
                            corrected['id'], 
                            corrected['code'], 
                            corrected
                        )
                        
                        if success:
                            self.log.correct("Correction successful!")
                            corrected['success'] = True
                            corrected['metrics'] = metrics
                            return corrected
                        else:
                            # error is already set from the evaluator return
                            pass
                            
                except Exception as e:
                    self.log.warning(f"Correction attempt failed: {e}")
                    error = str(e)
                    
        return None
    
    def _process_algorithm(self, algorithm: Dict[str, Any]) -> None:
        """Process a single algorithm through evaluation and analysis."""
        # Ensure algorithm has an ID
        if 'id' not in algorithm:
            algorithm['id'] = uuid.uuid4().hex
        self.log.process(f"Algorithm {algorithm['id']} (action: {algorithm.get('action', 'unknown')})")
        
        # Evaluate the algorithm
        success, metrics, error = self.evaluator.evaluate(
            algorithm['id'], 
            algorithm['code'], 
            algorithm
        )
        
        # Update algorithm with results
        algorithm['success'] = success
        algorithm['metrics'] = metrics
        if error:
            algorithm['error'] = error
        
        # Analyze results if successful
        if success:
            analysis = self.llm.analyze_results(
                prompt=self.config['prompts']['analysis'],
                algorithm=algorithm,
                history=self.history
            )
            algorithm['analysis'] = analysis
        
        # If failed, try to correct
        if not algorithm.get('success', False):
            error = algorithm.get('error', 'Unknown error')
            corrected = self._correct_algorithm(algorithm, error)
            
            if corrected:
                # Replace with corrected version
                algorithm = corrected
                algorithm['corrected'] = True
                self.stats.setdefault('total_corrections', 0)
                self.stats['total_corrections'] += 1
        
        # Add to history
        self.history.add(algorithm)
        
        # Update statistics
        if algorithm.get('success', False):
            self.stats['successful'] += 1
            metrics = algorithm.get('metrics', {})
            # Get performance classification
            perf_level = self.history._classify_performance(metrics)
            self.log.process(f"Success! Performance: {perf_level}", success=True)
        else:
            self.stats['failed'] += 1
            self.log.process(f"Failed: {algorithm.get('error', 'Unknown')[:100]}...", success=False)
    
    def _update_progress(self) -> None:
        """Update and display progress statistics."""
        total_evaluations = self.stats['successful'] + self.stats['failed']
        success_rate = (self.stats['successful'] / max(1, total_evaluations)) * 100
        
        self.log.info(
            f"Progress: {self.stats['total_generated']} generated, "
            f"{self.stats.get('total_corrections', 0)} corrections, "
            f"{success_rate:.1f}% success rate"
        )
        
        # Show best performance
        best = self.history.get_best_algorithm()
        if best:
            self.log.info(f"Best performance: {best['metrics']}")
    
    
    def _should_stop_early(self) -> bool:
        """Check if we should stop early based on performance."""
        # Check if we have enough excellent algorithms
        excellent_algos = self.history.get_by_performance('excellent')
        
        if len(excellent_algos) >= self.early_stop['target_excellent']:
            self.log.info(f"Found {len(excellent_algos)} excellent algorithms. Stopping early.")
            return True
            
        # Check if we have enough good + excellent algorithms
        good_algos = self.history.get_by_performance('good')
        if len(excellent_algos) + len(good_algos) >= self.early_stop['target_good_plus_excellent']:
            self.log.info(f"Found {len(excellent_algos)} excellent and {len(good_algos)} good algorithms. Stopping early.")
            return True
            
        # Also consider moderate algorithms in total count
        moderate_algos = self.history.get_by_performance('moderate')
        total_successful = len(excellent_algos) + len(good_algos) + len(moderate_algos)
        if total_successful >= self.early_stop['target_any_successful'] and (
            (len(excellent_algos) >= 1 if self.early_stop['require_at_least_one_excellent'] else True) or 
            (len(excellent_algos) + len(good_algos) >= self.early_stop['require_min_good_plus_excellent'])
        ):
            self.log.info(f"Found {total_successful} successful algorithms ({len(excellent_algos)} excellent, {len(good_algos)} good, {len(moderate_algos)} moderate). Stopping early.")
            return True
            
        return False
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the discovery session."""
        elapsed_time = time.time() - self.stats['start_time']
        
        # Performance distribution
        perf_dist = {
            'excellent': len(self.history.get_by_performance('excellent')),
            'good': len(self.history.get_by_performance('good')),
            'moderate': len(self.history.get_by_performance('moderate')),
            'poor': len(self.history.get_by_performance('poor'))
        }
        
        # Get best algorithms
        best_algorithms = self.history.get_top_performers(n=5)
        
        summary = {
            'name': self.name,
            'experiment_context': self.experiment_context,
            'statistics': {
                'total_generated': self.stats['total_generated'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': self.stats['successful'] / max(1, self.stats['successful'] + self.stats['failed']),
                'corrections': self.stats.get('total_corrections', 0),
                'duration_minutes': elapsed_time / 60
            },
            'performance_distribution': perf_dist,
            'best_algorithms': best_algorithms,
            'tuning_attempts': self.stats['tuning_attempts']
        }
        
        return summary
    
    def export_results(self, output_dir: str) -> None:
        """Export all results to the specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary = self._generate_summary()
        summary_file = output_path / 'discovery_summary.json'
        try:
            summary_file.write_text(json.dumps(summary, indent=2, allow_nan=False))
        except (TypeError, ValueError):
            summary_file.write_text(json.dumps(self._sanitize_for_json(summary), indent=2, allow_nan=False))
        
        # Export full history
        history_file = output_path / 'full_history.json'
        self.history.save(history_file)
        
        # Export best algorithms as separate files
        best_dir = output_path / 'best_algorithms'
        best_dir.mkdir(exist_ok=True)
        
        for idx, algo in enumerate(summary['best_algorithms'], start=1):
            perf_level = self.history._classify_performance(algo.get('metrics', {}))
            algo_file = best_dir / f"{idx:02d}_{perf_level}_{algo['id']}.py"
            
            # Create a complete Python file with metadata
            algo_content = f'''"""
Algorithm ID: {algo['id']}
Performance: {perf_level}
Metrics: {algo.get('metrics', {})}
Generated: {algo.get('timestamp', 'Unknown')}
Action: {algo.get('action', 'Unknown')}
"""

{algo['code']}
'''
            algo_file.write_text(algo_content)
        
        self.log.info(f"Results exported to {output_path}")
    
    def _load_checkpoint(self) -> None:
        """Load checkpoint if it exists."""
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint:
            self.log.checkpoint("Loaded checkpoint - resuming from previous session")
            self.stats = checkpoint.get('stats', self.stats)
            
            # Restore web search results if available
            self.web_search_results = checkpoint.get('web_search_results', None)
            if not self.web_search_results and getattr(self.llm, 'web_search_enabled', False):
                self.web_search_results = self.llm.web_search_context(
                    prompt=self.config['prompts']['web_search'],
                    user_context=self.experiment_context
                )
                self.web_search_results = str(self.web_search_results['results']) if self.web_search_results else ''
            
            # Reconstruct history from serialized data
            history_data = checkpoint.get('history', {})
            if isinstance(history_data, dict) and 'algorithms' in history_data:
                # Load algorithms back into history
                self.history.algorithms = history_data['algorithms']
                self.history._rebuild_indices()
            else:
                # Legacy format - handle gracefully
                self.log.warning("Legacy checkpoint format detected, starting with empty history")
                
            # Restore experiment context
            saved_context = checkpoint.get('experiment_context')
            if saved_context and not self.experiment_context:
                self.experiment_context = saved_context
                
            self.log.checkpoint(f"Resuming with {self.history.size()} algorithms")
    
    def _sanitize_for_json(self, obj):
        """Recursively convert non-serializable and NaN/Inf values to safe strings."""
        import math
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return "Infinity" if math.isinf(obj) else "NaN"
            return obj
        if isinstance(obj, (str, int, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        else:
            # Convert any other type to string
            return str(obj)
    
    def _save_checkpoint(self, force: bool = False) -> bool:
        """Save checkpoint periodically."""
        if force or self.stats['total_generated'] % self.checkpoint_manager.checkpoint_interval == 0:
            # Debug logging
            self.log.debug(f"Attempting checkpoint save (force={force}, total_generated={self.stats['total_generated']})")
            
            # Export history to a serializable format
            # Sanitize algorithms to ensure JSON serialization works
            sanitized_algorithms = [self._sanitize_for_json(algo) for algo in self.history.algorithms]
            
            history_data = {
                'algorithms': sanitized_algorithms,
                'metadata': {
                    'name': getattr(self.history, 'name', self.name),
                    'compression_threshold': getattr(self.history, 'compression_threshold', 50)  # Default if missing
                }
            }
            
            checkpoint_data = {
                'stats': self._sanitize_for_json(self.stats),
                'history': history_data,
                'experiment_context': str(self.experiment_context) if self.experiment_context else None,
                'web_search_results': str(getattr(self, 'web_search_results', None))
            }
            
            # Log what we're about to save
            self.log.debug(f"Checkpoint data: {len(history_data['algorithms'])} algorithms, stats: {self.stats}")
            
            return self.checkpoint_manager.save_checkpoint(checkpoint_data, force=force)
            # Checkpoint save logging is now handled by CheckpointManager itself
        return False
    
    def discover(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the discovery process.
        
        Args:
            max_iterations: Optional override for max attempts
            
        Returns:
            Summary of the discovery session
        """
        # Gather experiment context if not provided
        if not self.experiment_context:
            self.experiment_context = self.gather_experiment_context()
            
        max_attempts = max_iterations or self.max_attempts
        self.log.info(f"Starting discovery session with max {max_attempts} attempts")
        
        if self.verbose:
            self.log.discovery("Starting discovery session")
            self.log.discovery(f"Target: {self.n_target_algorithms} high-quality algorithms")
            self.log.discovery(f"Experiment: {self.experiment_context[:100]}..." if self.experiment_context else "No experiment context")

        # Do initial web search if this is a fresh start
        if self.history.size() == 0 and self.experiment_context:
            self.log.discovery("Performing web search...")
            web_results = self.llm.web_search_context(
                prompt=self.config['prompts']['web_search'],
                user_context=self.experiment_context
            )
            # Store only serializable string representation
            self.web_search_results = str(web_results['results']) if web_results else ''

        consecutive_errors = 0
        
        while self.stats['total_generated'] < max_attempts and not self._shutdown_requested:
            try:
                # Check for shutdown request
                if self._shutdown_requested:
                    self.log.info("[SHUTDOWN] Graceful shutdown requested")
                    break
                    
                # Decide action
                action = self._decide_next_action()
                
                # Execute action
                if action == 'tune':
                    algorithms = self._tune_algorithms()
                    # If no algorithms to tune, fall back to generation
                    if not algorithms:
                        self.log.info("No algorithms to tune, falling back to generation")
                        algorithms = [self._generate_algorithm()]
                elif action == 'evolve':
                    algorithms = self._evolve_algorithms()
                    # If no algorithms to evolve, fall back to generation
                    if not algorithms:
                        self.log.info("No algorithms to evolve, falling back to generation")
                        algorithms = [self._generate_algorithm()]
                else:  # generate
                    # Allow batched generation without changing LLM engine
                    batch = max(1, self.action_policy['generate_batch_size'])
                    algorithms = [self._generate_algorithm() for _ in range(batch)]
                
                # Check if we would exceed max_attempts
                if self.stats['total_generated'] + len(algorithms) > max_attempts:
                    remaining = max_attempts - self.stats['total_generated']
                    if remaining <= 0:
                        break
                    algorithms = algorithms[:remaining]
                    self.log.info(f"Trimming batch to {remaining} algorithms to stay within max_attempts limit")
                
                # Assign generation numbers and increment counter
                for algorithm in algorithms:
                    self.stats['total_generated'] += 1
                    algorithm['generation'] = self.stats['total_generated']
                
                # Evaluate each algorithm
                for algorithm in algorithms:
                    self._process_algorithm(algorithm)
                    
                # Update progress
                self._update_progress()
                
                # Save checkpoint
                self._save_checkpoint()
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Check early stopping
                if self._should_stop_early():
                    self.log.discovery("Early stopping criteria met!")
                    break
                    
            except KeyboardInterrupt:
                self.log.info("[INTERRUPT] Discovery interrupted by user")
                # Save checkpoint on interruption
                self._save_checkpoint(force=True)
                # Always show these messages, not just in verbose mode
                self.log.discovery("[DISCOVERY] Session interrupted! Progress has been saved.")
                self.log.discovery(f"[DISCOVERY] To resume, run with the same config and name: '{self.name}'")
                self.log.discovery(f"[DISCOVERY] Current progress: {self.stats['total_generated']} algorithms generated")
                raise
                
            except Exception as e:
                consecutive_errors += 1
                self.log.error(f"Error in discovery loop: {e}", exc_info=True)
                
                if consecutive_errors >= self.max_consecutive_errors:
                    # Try to save checkpoint before failing
                    checkpoint_saved = False
                    try:
                        checkpoint_saved = self._save_checkpoint(force=True)
                    except Exception as checkpoint_error:
                        self.log.error(f"Failed to save checkpoint: {checkpoint_error}")
                        
                        # Emergency backup - try to save algorithms individually
                        try:
                            emergency_dir = Path(self.config.get('save_path', './results')) / f"{self.name}_emergency_backup"
                            emergency_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save each algorithm as a separate file
                            for i, algo in enumerate(self.history.algorithms):
                                algo_file = emergency_dir / f"algorithm_{i}_{algo.get('id', 'unknown')}.json"
                                algo_file.write_text(json.dumps(self._sanitize_for_json(algo), indent=2))
                            
                            # Save stats
                            stats_file = emergency_dir / "stats.json"
                            stats_file.write_text(json.dumps(self._sanitize_for_json(self.stats), indent=2))
                            
                            self.log.warning(f"Emergency backup saved to: {emergency_dir}")
                        except Exception as e:
                            self.log.error(f"Emergency backup also failed: {e}")
                    
                    # Always show status, not just in verbose mode
                    if checkpoint_saved:
                        self.log.discovery(f"[DISCOVERY] Too many errors! Progress checkpoint saved.")
                        self.log.discovery(f"[DISCOVERY] To resume, run with the same config and name: '{self.name}'")
                    else:
                        self.log.discovery(f"[DISCOVERY] Too many errors! WARNING: Checkpoint save failed.")
                        self.log.discovery(f"[DISCOVERY] Check logs for emergency backup location or start from beginning.")
                    
                    raise GenerationError(f"Discovery loop failed after {consecutive_errors} consecutive errors")
                    
                # Continue with next iteration
                continue
        
        # Final checkpoint
        self._save_checkpoint(force=True)
        
        # Generate and return summary
        summary = self._generate_summary()
        
        if self.verbose:
            self.log.discovery("Discovery session complete!")
            self.log.discovery(f"Generated {summary['statistics']['total_generated']} algorithms")
            self.log.discovery(f"Success rate: {summary['statistics']['success_rate']:.1%}")
            self.log.discovery(f"Duration: {summary['statistics']['duration_minutes']:.1f} minutes")
        
        return summary
