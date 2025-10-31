# Ptychi‑Evolve

Automated discovery, evolution, and tuning of ptychographic reconstruction regularization functions using Large Language Models (LLMs). The framework integrates with the [`ptychi`](https://pty-chi.readthedocs.io/en/latest/) library, generates candidate regularization functions, executes and evaluates them, and maintains a history with compression and checkpointing for robust, iterative experimentation.

- LLM‑driven code generation, parameter tuning, error correction, and evolution
- Multi‑mode evaluation: ground truth, human‑in‑the‑loop, or VLM‑assisted
- LLM security analysis + restricted exec environment for running generated code
- History with performance classification, top‑N queries, and technical analysis


## Installation

```bash
conda create -n ptychi python=3.11
conda activate ptychi

pip install git+https://github.com/AdvancedPhotonSource/pty-chi@pear_aps
pip install hdf5plugin pyyaml backoff tifffile Pillow scipy scikit-image openai
```

Set OpenAI Credentials

```bash
export OPENAI_API_KEY="sk-..."
```


## Usage

Example Python script:

```python
import yaml
from ptychi_evolve.discovery import AlgorithmDiscovery
from ptychi_evolve.logging import setup_logging

setup_logging(
    log_file='discovery.log',
    log_level='DEBUG',
    console_level='INFO',
)

with open('path/to/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

experiment_context = "Short description of your experiment (sample, modality, challenges)."

discovery = AlgorithmDiscovery(config, experiment_context, verbose=True, debug=False)
summary = discovery.discover()            # run loop

# Export artifacts (summary + full history + best algorithms)
discovery.export_results('./results_dir')
```

Example discovery config yaml:

```yaml
name: my_experiment
max_attempts: 100
n_target_algorithms: 10
checkpoint_interval: 3
save_path: ./my_results

# ptychi options
reconstruction:
  data_directory: /abs/path/to/data
  scan_num: 1
  instrument: stem
  beam_source: electron
  beam_energy_kev: 300
  det_sample_dist_m: 1
  dk: 1/0.8/128
  diff_pattern_size_pix: 128
  diff_pattern_center_x: 64
  diff_pattern_center_y: 64
  load_processed_hdf5: true
  path_to_processed_hdf5_dp: /abs/path/to/data_roi0_dp.hdf5
  path_to_processed_hdf5_pos: /abs/path/to/data_roi0_para.hdf5
  number_of_slices: 1
  save_freq_iterations: 50

system:
  gpu_id: 0

evaluation:
  mode: ground_truth
  iterations: 1000
  ground_truth:
    object_path: /abs/path/to/ground_truth.tiff

llm:
  model: o4-mini
  reasoning_model: o3
  reasoning_effort: high
  search_context_size: high

search:
  enabled: true

analysis:
  performance_levels_ground_truth_label: ssim
  performance_levels_ground_truth_label_sense: higher_is_better
  performance_levels_ground_truth:
    excellent: 0.90
    good: 0.80
    moderate: 0.60
```

Additional knobs (optional):

```yaml
discovery:
  # Action selection policy
  action_policy:
    honor_suggestion_min_level: moderate   
    warmup_iterations: ${n_warmup_iterations}
    generate_batch_size: 1
    tune:
      min_excellent: 3
      every_k: 3
    evolve:
      min_successful: 5
      every_k: 2
    fallback_action: generate

  # Early stop policy
  early_stop:
    target_excellent: ${n_target_algorithms}
    target_good_plus_excellent: ${n_target_algorithms}
    target_any_successful: ${n_target_algorithms}
    require_min_good_plus_excellent: 5
    require_at_least_one_excellent: true

  # Error policy
  max_consecutive_errors: 5

history:
  compression_threshold: 50
  compression_moderate_share: 0.25
  compression_min_moderate: 5
  compression_keep_top_n: 3
  recent_for_context: 20
  suggestion_pool_last_k_high: 10
  recent_suggestions_last_k: 5
```

Major YAML keys explanations:

- `name`: session name used in history and checkpoints
- `reconstruction`: paths and parameters required by `ptychi`; e.g. data directory, scan number, instrument, detector geometry, iteration count, etc.
- `evaluation`:
  - `mode`: `ground_truth`, `human`, `few_shot`, `vision_description`
  - `iterations`: reconstruction iterations for evaluation
- `llm`: model choices, reasoning model/effort, and `search_context_size`
- `search.enabled`: enable initial web search
- `analysis`: performance thresholds and label configuration
- `history`: compression policy knobs and context sizes
- `discovery.action_policy`: knobs for action selection and batching
- `discovery.early_stop`: knobs for early stop behavior


Results & Artifacts:

- Summary JSON (high‑level stats, performance distribution, top algorithms)
- Full history JSON (all algorithms, metrics, analyses)
- Best algorithms as standalone `.py` files in `best_algorithms/`
- Checkpoints in `<save_path>/<name>_checkpoint.json` 


Using the best discovered regularizer:
```python
import ptychi.pear as pear
from ptychi.data_structures.object import PlanarObject

# Paste or import the discovered function here
def regularize_llm(self):
    # ... discovered code ...

# Enable the hook by monkey-patching
PlanarObject.regularize_llm = regularize_llm

params = {
    # ... your reconstruction params ...
    'object_regularization_llm': True,
}

task, recon_path, params = pear.ptycho_recon(run_recon=True, **params)
```

## How This Works

End‑to‑end loop:

1) Decide next action
- Uses heuristics and prior suggestions to choose between `generate`, `tune`, or `evolve`.
- Can honor evaluator’s suggested action from recent successful algorithms.

2) Produce algorithm(s)
- Generate new candidates with `prompts/discovery.md`.
- Tune parameters of top performers with `prompts/parameter_tuning.md`.
- Evolve algorithms via crossover/mutation with `prompts/evolution_*.md`.
- On evaluation error, attempt automatic correction with `prompts/algorithm_correction.md`.

3) Evaluate
- Monkey‑patches `PlanarObject.regularize_llm` with the generated function.
- Runs `ptychi.pear.ptycho_recon(...)` with `object_regularization_llm=True`.
- Scores output:
  - Ground truth mode: compute RMSE/MAE/SSIM/PSNR from saved TIFF outputs vs GT.
  - Human mode: interactive 0–1 metrics + qualitative feedback + suggested action.
  - VLM modes: few‑shot or description‑driven evaluation via OpenAI vision.

4) Analyze + Store
- LLM analyzes code/metrics into JSON: techniques, parameters, suggestions.
- Adds to history with classified performance and optional compression.
- Periodically checkpoint state for resume.

5) Stop when sufficient high‑quality algorithms are found or `max_attempts` reached.


Evaluation Modes:
- Ground truth
  - Metrics: RMSE/MAE/SSIM/PSNR on normalized phase images; warns and resizes GT if shapes mismatch
- Human
  - Interactive scoring (0–1 scale) and qualitative feedback; can abort evaluation
- VLM (few‑shot)
  - Provide representative images and structured evaluations; model learns the schema and scores new reconstructions
- VLM (vision description)
  - Supply natural language criteria; VLM produces structured evaluation; optionally confirm/override by human


Safety & Security:

- Pre‑execution LLM security audit flags system calls, file I/O, network, dangerous imports, sandbox escapes, and resource risks
- Generated code runs with a restricted `__builtins__` in a curated namespace with lazy heavy imports
- Keep `disable_security_scan: false` unless you fully trust the inputs and environment
- Always run in a controlled environment and review top candidates before reuse


Checkpointing & Resume:

- Saves checkpoints every `checkpoint_interval` algorithms (or on interrupt)
- On startup, automatically loads the last checkpoint and resumes progress (including experiment context and prior web search results)
- Writes an emergency backup if JSON serialization fails


Logging:

- Configure with `setup_logging(...)` for clean console + optional file
- Categories include `[INIT]`, `[DISCOVERY]`, `[GENERATE]`, `[EVAL]`, `[LLM]`, `[TUNE]`, `[EVOLVE]`, `[CHECKPOINT]`, `[DECISION]`


## License

This project is released under the MIT License. See `LICENSE` for details.
