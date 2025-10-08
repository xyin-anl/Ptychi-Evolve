"""
Evaluator for reconstruction algorithms.
Configurable evaluation with multiple modes (ground truth, human, VLM).
"""

import sys
import time
import logging
import torch
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import ast
import operator

import ptychi.pear as pear
from ptychi.data_structures.object import PlanarObject

# Import VLM engine if available
try:
    from .vlm_engine import VLMEngine
except ImportError:
    VLMEngine = None
    
from ptychi_evolve.logging import get_logger

# VLM availability will be checked during initialization

EVALUATION_CRITERIA = {
    'quality_score': 'Overall quality score (0.0-1.0, where 1.0 is perfect)',
    'noise_suppression': 'Noise suppression effectiveness (0.0-1.0)',
    'detail_preservation': 'Detail preservation (0.0-1.0)',
    'artifact_score': 'Free from artifacts (0.0-1.0, where 1.0 = no artifacts)',
    'physical_plausibility': 'Physical plausibility of the phase distribution (0.0-1.0)',
    'feedback': 'Feedback on the reconstruction quality (short text)',
}

def safe_eval_math(expr):
    """Safely evaluate a mathematical expression string.
    
    Only allows basic math operations and numbers.
    """
    if isinstance(expr, (int, float)):
        return expr
    
    if not isinstance(expr, str):
        return expr
    
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Num):  # Python 3.7 and earlier
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    
    try:
        tree = ast.parse(expr, mode='eval')
        return eval_node(tree.body)
    except Exception:
        # If evaluation fails, return the original value but warn
        logging.warning(f"safe_eval_math: failed to parse expression '{expr}', passing it through unchanged")
        return expr

class ReconEvaluator:
    """Evaluator for ptychographic reconstruction algorithms.
    
    Example evaluation configuration:
    
    ```python
    config = {
        'evaluation': {
            'mode': 'ground_truth',  # or 'human', 'few_shot', 'vision_description', 'auto'
            'iterations': 1000,
            'ground_truth': {
                'object_path': 'path/to/ground_truth.mat'
            },
            # For VLM modes
            'vlm': {
                'model': 'gpt-4.1'
            },
            'human_confirmation': True,  # For VLM modes
            'evaluation_description': 'Evaluate quality...',  # For vision_description mode
            'few_shot_examples': [  # For few_shot mode
                {
                    'image_path': 'example1.tiff',
                    'evaluation': {
                        'quality_score': 0.9,
                        'noise_suppression': 0.8,
                        ...
                    }
                }
            ]
        },
    }
    ```
    """
    
    def __init__(self, config: Dict[str, Any], llm_engine=None, verbose: bool = False, debug: bool = False):
        """Initialize evaluator with configuration."""
        if not torch.cuda.is_available():
            # CUDA warning will be handled by ptychi library
            pass
        self.config = config
        self.eval_config = config['evaluation']
        self.llm_engine = llm_engine  # Store for security analysis
        self.verbose = verbose
        self.debug = debug
        
        # Initialize unified logger
        self.log = get_logger(__name__, verbose=self.verbose, debug=self.debug)
        
        # Base reconstruction parameters from config
        self.base_params = self._load_base_params()
        
        # Evaluation mode: 'ground_truth', 'human', 'few_shot', 'vision_description', 'auto'
        self.eval_mode = self.eval_config.get('mode', 'human')
        self.ground_truth_available = self._check_ground_truth()
        
        self.log.eval(f"Initialized with mode: {self.eval_mode}")
        self.log.eval(f"Ground truth available: {self.ground_truth_available}")
        self.log.eval(f"Iterations per evaluation: {self.eval_config.get('iterations', 500)}")

        # Reproducibility: optional seeding
        try:
            import random as _random
            import numpy as _np
            seed = self.config.get('system', {}).get('seed')
            if seed is not None:
                _random.seed(seed)
                _np.random.seed(seed)
                try:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    if self.config.get('system', {}).get('deterministic_torch', False):
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
        except Exception:
            pass

        # Evaluation criteria (human/VLM) from config with defaults
        criteria_cfg = self.config.get('evaluation', {}).get('criteria', {})
        self.evaluation_criteria = {
            **EVALUATION_CRITERIA,
            **criteria_cfg,
        }
        
        # Handle auto mode
        if self.eval_mode == 'auto':
            self.eval_mode = self._determine_auto_mode()
            self.log.eval(f"Auto mode selected: {self.eval_mode}")
        
        # Validate selected mode
        if self.eval_mode == 'ground_truth' and not self.ground_truth_available:
            self.log.warning("Ground truth is not available for ground truth mode evaluation, falling back to human mode")
            self.eval_mode = 'human'
        
        # Initialize VLM engine if needed and available
        self.vlm_engine = None
        if self.eval_mode in ['few_shot', 'vision_description']:
            assert VLMEngine is not None, "VLM engine is not available for {} mode evaluation".format(self.eval_mode)
            # Pass full config to VLM engine
            self.vlm_engine = VLMEngine(config)
            
            # Load few-shot examples if in few-shot mode
            if self.eval_mode == 'few_shot':
                self._load_few_shot_examples()
    
    def _check_ground_truth(self) -> bool:
        """Check if ground truth is available."""
        gt_config = self.eval_config.get('ground_truth', {})
        if not gt_config:
            return False
            
        gt_path = gt_config.get('object_path') 
        if not gt_path:
            return False
            
        return Path(gt_path).exists()
    
    def _determine_auto_mode(self) -> str:
        """Automatically determine the best evaluation mode based on available resources."""
        # Priority order:
        # 1. Ground truth (if available)
        if self.ground_truth_available:
            return 'ground_truth'
            
        # 2. Few-shot VLM (if examples and VLM available)
        if VLMEngine is not None and self.eval_config.get('few_shot_examples'):
            return 'few_shot'
            
        # 3. Vision description VLM (if description and VLM available)
        if VLMEngine is not None and self.eval_config.get('evaluation_description'):
            return 'vision_description'
            
        # 4. Default to human
        return 'human'
    
    def _load_few_shot_examples(self):
        """Load few-shot examples for VLM evaluation."""
        few_shot_config = self.eval_config.get('few_shot_examples', [])
        
        for example in few_shot_config:
            image_path = Path(example['image_path'])
            if not image_path.exists():
                self.log.warning(f"Few-shot example image not found: {image_path}")
                continue
            
            # Support both structured and natural language evaluations
            if 'evaluation' in example and isinstance(example['evaluation'], str):
                # Natural language evaluation
                evaluation = {'feedback': example['evaluation']}
            elif 'evaluation' in example and isinstance(example['evaluation'], dict):
                # Structured evaluation 
                evaluation = {}
                for key in self.evaluation_criteria.keys():
                    if key in example['evaluation']:
                        evaluation[key] = example['evaluation'][key]
            else:
                raise ValueError(f"Invalid few-shot example format: {example}")
            
            self.vlm_engine.add_few_shot_example(image_path, evaluation)
            self.log.info(f"Loaded few-shot example: {image_path}")
        
        
    def _load_base_params(self) -> Dict[str, Any]:
        """Load base reconstruction parameters from config."""
        recon_config = self.config.get('reconstruction', {})
        system_config = self.config.get('system', {})
        
        # Check for required fields and provide helpful error messages
        required_fields = ['data_directory', 'scan_num', 'instrument']
        missing_fields = [field for field in required_fields if field not in recon_config]
        if missing_fields:
            raise ValueError(f"Missing required reconstruction config fields: {', '.join(missing_fields)}")
        
        params = {
            'data_directory': recon_config['data_directory'],
            'scan_num': recon_config['scan_num'],
            'instrument': recon_config['instrument'],
            'beam_source': recon_config.get('beam_source', 'xray'),
            'beam_energy_kev': recon_config.get('beam_energy_kev', 8.0),
            'det_sample_dist_m': recon_config.get('det_sample_dist_m', 0.5),
            'dk': safe_eval_math(recon_config.get('dk')),  # Critical for object size - evaluates math expressions
            'diff_pattern_size_pix': recon_config.get('diff_pattern_size_pix', 256),
            'diff_pattern_center_x': recon_config.get('diff_pattern_center_x', 128),
            'diff_pattern_center_y': recon_config.get('diff_pattern_center_y', 128),
            'load_processed_hdf5': recon_config.get('load_processed_hdf5', False),
            'path_to_processed_hdf5_dp': recon_config.get('path_to_processed_hdf5_dp', ''),
            'path_to_processed_hdf5_pos': recon_config.get('path_to_processed_hdf5_pos', ''),
            'transpose_diffraction_patterns': recon_config.get('transpose_diffraction_patterns', False),
            'path_to_init_positions': recon_config.get('path_to_init_positions', ''),
            'path_to_init_probe': recon_config.get('path_to_init_probe', ''),
            'path_to_init_object': recon_config.get('path_to_init_object', ''),
            'use_model_FZP_probe': recon_config.get('use_model_FZP_probe', False),
            'init_probe_propagation_distance_mm': recon_config.get('init_probe_propagation_distance_mm', 0.0),
            'orthogonalize_initial_probe': recon_config.get('orthogonalize_initial_probe', False),
            'position_correction': recon_config.get('position_correction', False),
            'position_correction_update_limit': recon_config.get('position_correction_update_limit', 10.0),
            'position_correction_affine_constraint': recon_config.get('position_correction_affine_constraint', False),
            'intensity_correction': recon_config.get('intensity_correction', False),
            'center_probe': recon_config.get('center_probe', False),
            'probe_support': recon_config.get('probe_support', None),
            'number_probe_modes': recon_config.get('number_probe_modes', 1),
            'update_object_w_higher_probe_modes': recon_config.get('update_object_w_higher_probe_modes', True),
            'number_opr_modes': recon_config.get('number_opr_modes', 1),
            'update_batch_size': recon_config.get('update_batch_size'),
            'number_of_batches': recon_config.get('number_of_batches', 1),
            'batch_selection_scheme': recon_config.get('batch_selection_scheme', 'random'),
            'momentum_acceleration': recon_config.get('momentum_acceleration', False),
            'number_of_slices': recon_config.get('number_of_slices', 1),
            'object_thickness_m': float(recon_config.get('object_thickness_m', 1e-6)),
            'layer_regularization': recon_config.get('layer_regularization', False),
            'position_correction_layer': recon_config.get('position_correction_layer'),
            'save_freq_iterations': recon_config.get('save_freq_iterations', 100),
            'recon_dir_suffix': recon_config.get('recon_dir_suffix', 'llm'),
            'recon_parent_dir': recon_config.get('recon_parent_dir', ''),
            'gpu_id': system_config.get('gpu_id', 0),
            'save_diffraction_patterns': recon_config.get('save_diffraction_patterns', False),
            'object_regularization_llm': True  # Always true for LLM regularization
        }
        
        return params
    
    
    def _create_function(self, code: str):
        """Create regularization function from code with security checks."""
        # Security check if LLM engine is available
        if self.llm_engine and not self.config.get('disable_security_scan', False):
            security_result = self.llm_engine.analyze_code_security(
                code, 
                context="Ptychography regularization function for tensor operations"
            )
            
            if not security_result.get('is_safe', False):
                issues = security_result.get('issues', ['Unknown security issue'])
                raise ValueError(f"Security check failed: {'; '.join(issues)}")
        
        # Create namespace for exec with lazy imports
        # Restrict builtins to whitelist for security
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'complex': complex, 'dict': dict, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'int': int, 'len': len,
            'list': list, 'map': map, 'max': max, 'min': min,
            'pow': pow, 'range': range, 'round': round, 'set': set,
            'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
            'zip': zip, 'isinstance': isinstance, 'hasattr': hasattr,
            'getattr': getattr, 'setattr': setattr, 'type': type,
            'print': print,  # Allow print for debugging
            # Additional commonly used built-ins
            'object': object, 'Exception': Exception, 'slice': slice,
            'property': property, 'staticmethod': staticmethod,
            'classmethod': classmethod, 'callable': callable,
            'iter': iter, 'next': next, 'reversed': reversed,
            'ValueError': ValueError, 'TypeError': TypeError,
            'RuntimeError': RuntimeError, 'AttributeError': AttributeError,
            'IndexError': IndexError, 'KeyError': KeyError,
            'NotImplementedError': NotImplementedError,
        }
        
        # Limited import whitelist for generated code
        ALLOWED_IMPORTS = {
            'math','random','numpy','np','torch','scipy','skimage',
            'scipy.ndimage','collections','itertools','functools','numbers',
            'typing'
        }
        _orig_import = __import__
        def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split('.')[0]
            if name not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                raise ImportError(f"Import of '{name}' is not allowed in LLM regularizer.")
            return _orig_import(name, globals, locals, fromlist, level)
        safe_builtins['__import__'] = _limited_import

        namespace = {
            '__builtins__': safe_builtins,
            'torch': torch,
            'np': np,
            'numpy': np,
            'F': torch.nn.functional,
            'math': _limited_import('math'),
            'random': _limited_import('random'),
            'numbers': _limited_import('numbers'),
            'typing': _limited_import('typing'),
            'collections': _limited_import('collections'),
            'itertools': _limited_import('itertools'),
            'functools': _limited_import('functools'),
        }
        
        # Add lazy imports for heavy libraries
        def lazy_scipy():
            return _limited_import('scipy')
        
        def lazy_skimage():
            return _limited_import('skimage')
        
        def lazy_ndimage():
            return _limited_import('scipy.ndimage', fromlist=['ndimage'])
        
        # Use property-like access for lazy loading
        class LazyModule:
            def __init__(self, loader, module_name):
                self._loader = loader
                self._module = None
                self._module_name = module_name
                
            def __getattr__(self, name):
                if self._module is None:
                    try:
                        self._module = self._loader()
                        # Populate sys.modules to prevent double imports
                        sys.modules[self._module_name] = self._module
                    except ImportError as e:
                        raise ImportError(f"Failed to import {self._module_name} required by algorithm: {e}")
                return getattr(self._module, name)
        
        namespace['scipy'] = LazyModule(lazy_scipy, 'scipy')
        namespace['skimage'] = LazyModule(lazy_skimage, 'skimage')
        namespace['ndimage'] = LazyModule(lazy_ndimage, 'scipy.ndimage')
        
        exec(code, namespace)
        
        if 'regularize_llm' not in namespace:
            raise ValueError("Code must define 'regularize_llm' function")
            
        return namespace['regularize_llm']

    def _remove_function(self, original_method):
        """Remove the function from the PlanarObject class."""
        if original_method:
            PlanarObject.regularize_llm = original_method
        else:
            if hasattr(PlanarObject, 'regularize_llm'):
                delattr(PlanarObject, 'regularize_llm')
    
    def _compute_phase_metrics(self, rec_phase_path: Union[str, Path]) -> Dict[str, float]:
        """Compute phase reconstruction metrics against ground truth.
        
        Args:
            rec_phase_path: Path to the reconstructed phase TIFF file saved by pear_io
        
        Returns:
            Dictionary of metrics
        """
        import tifffile
        import scipy.io
        from scipy.ndimage import zoom
        
        # Load reconstructed phase from TIFF (saved by pear_io)
        # This is already unwrapped and normalized to 16-bit
        rec_phase_normalized = tifffile.imread(rec_phase_path)
        
        # Convert from 16-bit normalized back to phase values
        # Note: We lose the absolute phase offset during normalization, 
        # so we'll normalize both rec and GT for fair comparison
        rec_phase = rec_phase_normalized.astype(np.float32) / 65535.0
        
        # Load ground truth
        gt_path = self.eval_config['ground_truth']['object_path']
        if gt_path.endswith('.mat'):
            gt_data = scipy.io.loadmat(gt_path)
            gt_phase = gt_data.get('phase', gt_data.get('object_phase', None))
            if gt_phase is None:
                raise ValueError(f"Could not find phase data in {gt_path}")
            gt_phase = gt_phase.astype(np.float32)
        elif gt_path.endswith('.tiff') or gt_path.endswith('.tif'):
            gt_phase = tifffile.imread(gt_path)
            gt_phase = gt_phase.astype(np.float32)
        else:
            raise ValueError(f"Unsupported ground truth format: {gt_path}")
        
        # Ensure same shape
        if gt_phase.shape != rec_phase.shape:
            warning_msg = (
                f"\n" + "="*80 + "\n"
                f"WARNING: Shape mismatch between ground truth and reconstruction!\n"
                f"Ground truth shape: {gt_phase.shape}\n"
                f"Reconstructed shape: {rec_phase.shape}\n"
                f"\n"
                f"Ground truth will be rescaled using zoom interpolation.\n"
                f"This WILL AFFECT metric accuracy and make results unreliable!\n"
                f"\n"
                f"RECOMMENDED: Use pre-aligned data with matching dimensions.\n"
                + "="*80 + "\n"
            )
            self.log.warning(warning_msg)
            zoom_factors = [r/g for r, g in zip(rec_phase.shape, gt_phase.shape)]
            gt_phase = zoom(gt_phase, zoom_factors, order=1)
        
        # Normalize to [0,1] robustly (avoid divide-by-zero for constant images)
        def _normalize01(a: np.ndarray) -> np.ndarray:
            a = a.astype(np.float32)
            amin = float(a.min())
            amax = float(a.max())
            rng = amax - amin
            if rng == 0:
                return np.zeros_like(a, dtype=np.float32)
            return (a - amin) / rng

        gt_phase_norm = _normalize01(gt_phase)
        rec_phase_norm = _normalize01(rec_phase)
        
        # Compute metrics on normalized phases
        metrics = {
            'rmse': float(np.sqrt(np.mean((gt_phase_norm - rec_phase_norm)**2))),
            'mae': float(np.mean(np.abs(gt_phase_norm - rec_phase_norm))),
            'ssim': self._compute_ssim(gt_phase_norm, rec_phase_norm),
            'psnr': self._compute_psnr(gt_phase_norm, rec_phase_norm)
        }
            
        return metrics
  
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute SSIM between two images."""
        from skimage.metrics import structural_similarity
        # Use a reasonable, valid odd window size (>=3)
        min_dim = min(img1.shape)
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(win_size, 3)
        return float(structural_similarity(img1, img2, data_range=img1.max() - img1.min(), win_size=win_size))
    
    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute PSNR between two images."""
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            # Cap PSNR to avoid Infinity in downstream JSON serialization
            return 120.0
        # For normalized images [0,1], max_val is always 1.0
        max_val = 1.0
        return float(20 * np.log10(max_val / np.sqrt(mse)))

    def _calculate_metrics_ground_truth(self, phase_path: str) -> Dict[str, Any]:
        """Calculate metrics from reconstruction task using saved files."""
        assert self.ground_truth_available, "Ground truth is not available"
        metrics = {}
        
        try:
            # Compute phase metrics
            phase_metrics = self._compute_phase_metrics(phase_path)
            metrics.update(phase_metrics)

        except Exception as e:
            self.log.error(f"Failed to calculate metrics: {e}")
            metrics['metrics_error'] = str(e)
            
        return metrics

    def _collect_human_evaluation(self, image_path: Union[str, Path], final_loss: float = None) -> Dict[str, Any]:
        """Collect structured human evaluation of reconstruction quality."""
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Human evaluation requested but no interactive TTY is available. "
                "Use ground_truth or VLM modes, or run interactively."
            )
        self.log.eval("HUMAN EVALUATION OF RECONSTRUCTION QUALITY")
        
        if final_loss is not None:
            self.log.eval(f"Final Loss: {final_loss}")
        self.log.eval(f"Reconstruction image saved at: {image_path}")
        self.log.eval("Please review the reconstructed phase image and provide your evaluation.")
        
        self.log.eval("Please evaluate the reconstruction based on the following criteria:")

        metrics = {}
        
        for key, description in self.evaluation_criteria.items():
            if key == 'feedback':
                continue
            
            while True:
                try:
                    sys.stdout.flush()
                    user_input = input(f"{description} (or 'q' to quit): ")
                    if user_input.lower() == 'q':
                        self.log.eval("Evaluation aborted by user.")
                        return {'aborted': True}
                    score = float(user_input)
                    if 0 <= score <= 1:
                        metrics[key] = score
                        break
                    else:
                        self.log.eval("Please enter a value between 0.0 and 1.0")
                except ValueError:
                    self.log.eval("Please enter a valid number or 'q' to quit")

        # Qualitative feedback
        self.log.eval("Please provide qualitative feedback:")
        self.log.eval("What aspects of the reconstruction work well? What needs improvement?")
        sys.stdout.flush()
        feedback = input("Your feedback: ") 
        if feedback:
            metrics['feedback'] = feedback
        
        # Ask for suggested action for discovery process
        self.log.eval("Based on this evaluation, what should the discovery system do next?")
        self.log.eval("Options: generate (new algorithm), tune (parameters), evolve (combine algorithms)")
        sys.stdout.flush()
        suggested_action = input("Suggested action [generate/tune/evolve]: ").lower().strip()
        if suggested_action in ['generate', 'tune', 'evolve']:
            metrics['suggested_action'] = suggested_action
         
        return metrics
 
    
    def _calculate_metrics_human(self, phase_path: str, final_loss: float = None) -> Dict[str, Any]:
        """Obtain metrics for human evaluation mode."""
        metrics = {}
        
        try:
            # Get human evaluation of reconstruction quality
            human_metrics = self._collect_human_evaluation(phase_path, final_loss)
            # Store as structured evaluation for consistency
            metrics['structured_evaluation'] = human_metrics
            # Also store suggested action at top level for discovery
            if 'suggested_action' in human_metrics:
                metrics['suggested_action'] = human_metrics['suggested_action']
                
        except Exception as e:
            self.log.error(f"Failed to calculate human evaluation metrics: {e}")
            metrics['metrics_error'] = str(e)
            
        return metrics

    def _confirm_vlm_evaluation(self, vlm_results: Dict[str, Any], image_path: Union[str, Path]) -> Dict[str, Any]:
        """Confirm or modify VLM's structured evaluation."""
        self.log.eval("VLM EVALUATION RESULTS - CONFIRMATION REQUIRED")
        
        metrics = vlm_results.copy()
        
        # Show VLM's structured evaluation
        self.log.eval("VLM Structured Evaluation:")
        for key, value in metrics.items():
            if key != 'raw_text':  # Skip raw text field
                self.log.eval(f"- {key}: {value}")
        
        # Ask for confirmation
        sys.stdout.flush()
        confirm = input("Do you agree with this evaluation? (y/n): ").lower()
        
        if confirm != 'y':
            self.log.prompt("Enter human override:")
            human_metrics = self._collect_human_evaluation(image_path)
            return human_metrics
        else:
            # Ask for suggested action even if accepting VLM evaluation
            self.log.eval("Based on this evaluation, what should the discovery system do next?")
            self.log.eval("Options: generate (new algorithm), tune (parameters), evolve (combine algorithms)")
            sys.stdout.flush()
            suggested_action = input("Suggested action [generate/tune/evolve]: ").lower().strip()
            if suggested_action in ['generate', 'tune', 'evolve']:
                metrics['suggested_action'] = suggested_action
        
        return metrics
  
    
    def _calculate_metrics_vlm_few_shot(self, phase_path: str, 
                                     algorithm_info: Dict[str, Any], 
                                     final_loss: float = None) -> Dict[str, Any]:
        """Extract metrics using VLM few-shot learning mode."""
        metrics = {}
        
        try:
            # Get VLM evaluation
            require_confirmation = self.eval_config.get('human_confirmation', True)
            vlm_results = self.vlm_engine.evaluate_with_few_shot(Path(phase_path))
            
            # Process the VLM structured evaluation
            if require_confirmation:
                # Present VLM evaluation to human for confirmation
                confirmed_results = self._confirm_vlm_evaluation(vlm_results, phase_path)
                metrics['structured_evaluation'] = confirmed_results
                # Extract suggested action if present
                if 'suggested_action' in confirmed_results:
                    metrics['suggested_action'] = confirmed_results['suggested_action']
            else:
                metrics['structured_evaluation'] = vlm_results

        except Exception as e:
            self.log.error(f"Failed to calculate VLM few-shot metrics: {e}")
            metrics['metrics_error'] = str(e)
            
        return metrics


    def _calculate_metrics_vlm_description(self, phase_path: str,
                                       algorithm_info: Dict[str, Any],
                                       final_loss: float = None) -> Dict[str, Any]:
        """Extract metrics using VLM with natural language description."""
        assert 'evaluation_description' in self.eval_config, "Evaluation description is not available"
        assert self.eval_config['evaluation_description'] != '', "Evaluation description is empty"
        assert self.eval_config['evaluation_description'] != 'N/A', "Evaluation description is N/A"
        assert self.eval_config['evaluation_description'] != None, "Evaluation description is None"
        
        metrics = {}
        
        try:
            # Get evaluation description
            description = self.eval_config['evaluation_description']
            
            # Get VLM evaluation
            require_confirmation = self.eval_config.get('human_confirmation', True)
            vlm_results = self.vlm_engine.evaluate_with_description(
                Path(phase_path), description
            )
            
            if require_confirmation:
                # Present VLM evaluation to human for confirmation
                self.log.eval("VLM EVALUATION RESULTS - VISION DESCRIPTION MODE")
                
                self.log.eval(f"Image: {phase_path}")
                self.log.eval(f"Evaluation Criteria Applied:")
                self.log.eval("-" * 40)
                self.log.eval(description)
                self.log.eval("-" * 40)
                
                self.log.eval(f"VLM Quality Score: {vlm_results.get('quality_score', 'N/A')}/1.0")
                self.log.eval(f"VLM Feedback: {vlm_results.get('feedback', 'N/A')}")
                
                # Ask for confirmation
                sys.stdout.flush()
                confirm = input("Do you agree with this evaluation? (yes/no): ").lower()
                
                if confirm.startswith('n'):
                    self.log.prompt("Enter human override:")
                    human_metrics = self._collect_human_evaluation(phase_path, final_loss)
                    metrics['structured_evaluation'] = human_metrics
                    # Extract suggested action if present
                    if 'suggested_action' in human_metrics:
                        metrics['suggested_action'] = human_metrics['suggested_action']
                else:
                    metrics['structured_evaluation'] = vlm_results
            else:
                metrics['structured_evaluation'] = vlm_results
            
        except Exception as e:
            self.log.error(f"Failed to extract VLM description metrics: {e}")
            metrics['metrics_error'] = str(e)
            
        return metrics

   
    def evaluate(self, algo_id: str, code: str, algorithm_data: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Evaluate algorithm through a single evaluation run.
        
        Args:
            algo_id: Algorithm identifier
            code: Regularization function code
            algorithm_data: Additional algorithm metadata (techniques, description, etc.)
            
        Returns:
            (success, metrics, error_message)
        """
        # Create regularization function
        try:
            self.log.eval("Creating regularization function from code")
            if self.debug:
                self.log.debug_info(f"[DEBUG] Evaluating algorithm {algo_id}")
                self.log.debug_info(f"Code to evaluate ({len(code)} chars):", 1)
                # Show more of the code
                code_lines = code.split('\n')[:20]
                for line in code_lines:
                    self.log.debug_info(f"{line}", 2)
                if len(code.split('\n')) > 20:
                    remaining_lines = len(code.split('\n')) - 20
                    self.log.debug_info(f"... ({remaining_lines} more lines)", 2)
                    
            regularize_func = self._create_function(code)
        except Exception as e:
            error_msg = f"Failed to create function: {str(e)}"
            self.log.error(error_msg)
            self.log.eval(f"Failed to create function: {str(e)[:100]}...")
            return False, {}, error_msg
 
        # Get evaluation parameters
        iterations = self.eval_config.get('iterations', 500)

        self.log.info(f"Running evaluation for {algo_id} with {iterations} iterations")
        
        self.log.eval(f"Running reconstruction with {iterations} iterations")

        # Store original method
        original_method = getattr(PlanarObject, 'regularize_llm', None)
        
        try:
            # Directly monkey-patch the generated function
            PlanarObject.regularize_llm = regularize_func
            
            # Setup parameters for this evaluation
            params = self.base_params.copy()
            params['number_of_iterations'] = iterations

            # Run reconstruction
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            start_time = time.time()
            self.log.eval("Starting ptychographic reconstruction...")
            self.log.eval(f"Data: {params.get('data_directory')}, Scan: {params.get('scan_num')}")
            result = pear.ptycho_recon(run_recon=True, **params)
            # Handle different return value counts gracefully
            if isinstance(result, tuple):
                if len(result) >= 2:
                    task = result[0]
                    recon_path = result[1]
                else:
                    raise ValueError(f"pear.ptycho_recon returned unexpected tuple length: {len(result)}")
            else:
                raise ValueError(f"pear.ptycho_recon returned unexpected type: {type(result)}")
            recon_time = time.time() - start_time
            self.log.info(f"Reconstruction time: {recon_time} seconds")

            # Calculate loss history
            final_loss = None  # Initialize to None to avoid UnboundLocalError
            if hasattr(task.reconstructor, 'loss_tracker'):
                loss_table = task.reconstructor.loss_tracker.table
                if loss_table is not None and len(loss_table) > 0:
                    loss_history = loss_table['loss'].tolist()
                    final_loss = float(loss_table['loss'].iloc[-1])
                    self.log.info(f"Final loss: {final_loss}")
            
            # Find reconstructed phase from saved TIFF files
            # Check if this is a multislice reconstruction
            phase_total_dir = Path(recon_path) / 'object_ph_total'
            phase_single_dir = Path(recon_path) / 'object_ph'
            
            if phase_total_dir.exists():
                # Multislice reconstruction - use total phase
                phase_dir = phase_total_dir
                phase_files = list(phase_dir.glob('object_ph_total_Niter*.tiff')) + \
                              list(phase_dir.glob('object_ph_total_Niter*.tif'))
            else:
                # Single slice reconstruction
                phase_dir = phase_single_dir
                phase_files = list(phase_dir.glob('object_ph_Niter*.tiff')) + \
                              list(phase_dir.glob('object_ph_Niter*.tif'))
                
            if not phase_files:
                raise ValueError(f"No phase files found in {phase_dir}")
            
            # Sort by iteration number and get the latest
            def _iter_from_name(p: Path) -> int:
                try:
                    return int(p.stem.split('Niter')[-1])
                except Exception:
                    return -1
            phase_path = max(phase_files, key=_iter_from_name)
            

            # Calculate recon quality metrics based on evaluation mode
            self.log.eval(f"Calculating metrics using {self.eval_mode} mode")
            
            if self.debug:
                self.log.debug_info("[DEBUG] Reconstruction results:")
                self.log.debug_info(f"Final loss: {final_loss}", 1)
                self.log.debug_info(f"Phase image path: {phase_path}", 1)
                self.log.debug_info(f"Reconstruction time: {recon_time:.2f}s", 1)
                self.log.debug_info(f"Reconstruction directory: {recon_path}", 1)
            
            if self.eval_mode == 'ground_truth':
                eval_metrics = self._calculate_metrics_ground_truth(phase_path)
            elif self.eval_mode == 'human':
                eval_metrics = self._calculate_metrics_human(phase_path, final_loss)
            elif self.eval_mode == 'few_shot':
                eval_metrics = self._calculate_metrics_vlm_few_shot(phase_path, algorithm_data or {}, final_loss)
            elif self.eval_mode == 'vision_description':
                eval_metrics = self._calculate_metrics_vlm_description(phase_path, algorithm_data or {}, final_loss)
            else:
                raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")
                
            if self.debug and eval_metrics:
                self.log.debug_info("[DEBUG] Evaluation metrics:")
                for key, value in eval_metrics.items():
                    if key != 'structured_evaluation' and not key.startswith('_'):
                        self.log.debug_info(f"{key}: {value}", 1)
                if 'structured_evaluation' in eval_metrics:
                    self.log.debug_info("Structured evaluation:", 1)
                    for k, v in eval_metrics['structured_evaluation'].items():
                        if not k.startswith('_'):
                            self.log.debug_info(f"{k}: {v}", 2)
            
            # Clean up GPU resources now that we're done with reconstruction
            del task
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Check if human evaluation was aborted
            if eval_metrics.get('structured_evaluation', {}).get('aborted', False):
                return False, eval_metrics, "Evaluation aborted by user"
            
            return True, eval_metrics, None
            
        except Exception as e:
            error_msg = f"Evaluation error: {str(e)}\n{traceback.format_exc()}"
            self.log.error(error_msg)
            self.log.eval(f"Evaluation failed: {str(e)[:100]}...")
            
            return False, {}, error_msg
        
        finally:
            self._remove_function(original_method)
