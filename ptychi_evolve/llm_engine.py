"""
LLM Engine for algorithm discovery.
Powers all text processing, code generation, and analysis operations.
Uses OpenAI Responses API with web search capabilities.
"""
import re
import json
import time
from typing import Dict, Any, List, Optional, Union
import openai
from openai import OpenAI
import backoff
from ptychi_evolve.history import DiscoveryHistory
from ptychi_evolve.logging import get_logger



class LLMEngine:
    """Engine for all LLM-powered operations in the discovery process."""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False, debug: bool = False):
        """Initialize LLM interface with configuration."""
        self.config = config
        self.llm_config = config.get('llm', {})
        self.verbose = verbose
        self.debug = debug
        
        # Initialize logger
        self.log = get_logger(__name__, verbose=self.verbose, debug=self.debug)
        
        # Model configuration
        self.model = self.llm_config.get('model', 'gpt-4o-mini')
        self.reasoning_model = self.llm_config.get('reasoning_model', 'o4-mini')

        # Reasoning models configuration
        self.reasoning_effort = self.llm_config.get('reasoning_effort', 'medium')
        
        self.log.llm(f"Initialized with model: {self.model}")
        self.log.llm(f"Reasoning model: {self.reasoning_model}")
        self.log.llm(f"Web search: {'enabled' if config.get('search', {}).get('enabled', True) else 'disabled'}")
        
        # Setup OpenAI client
        self.client = OpenAI(timeout=600)
        
        # Web search configuration
        self.web_search_enabled = config.get('search', {}).get('enabled', True)
        self.search_context_size = self.llm_config.get('search_context_size', 'medium')
        
        # Store prompts for crossover fallback
        self.prompts = config.get('prompts', {})
        
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _call_llm(self, 
                  input_content: Union[str, List[Dict]], 
                  instructions: Optional[str] = None,
                  tools: Optional[List[Dict]] = None,
                  use_reasoning: bool = False,
                  json_mode: bool = False,
                  force_web_search: Optional[bool] = None,
                  disable_web_search: bool = False) -> Dict[str, Any]:
        """Make LLM API call using Responses API with web search support."""
        try:
            # Build request parameters for Responses API
            params = {
                "model": self.model if not use_reasoning else self.reasoning_model,
                "input": input_content,  # Can be string or list of messages
            }
            
            # Add instructions if provided
            if instructions:
                params["instructions"] = instructions
                
            # Add tools (web search, etc.)
            # Note: Web search cannot be used with JSON mode
            if tools:
                params["tools"] = tools
            elif not disable_web_search and not json_mode:
                # Determine if web search should be used
                use_search = force_web_search if force_web_search is not None else self.web_search_enabled
                if use_search:
                    web_search_tool = {"type": "web_search_preview"}
                    # Add search context size if specified
                    if self.search_context_size:
                        web_search_tool["search_context_size"] = self.search_context_size
                    params["tools"] = [web_search_tool]
                
            # Add reasoning for supported models
            if use_reasoning:
                params["reasoning"] = {"effort": self.reasoning_effort}
            
            if json_mode:
                params["text"] = {"format": {"type": "json_object"}}
                
            self.log.debug_info("LLM API Call:")
            self.log.debug_info(f"Model: {params['model']}", 1)
            self.log.debug_info(f"Reasoning: {use_reasoning}", 1)
            self.log.debug_info(f"JSON mode: {json_mode}", 1)
            self.log.debug_info(f"Has tools: {bool(params.get('tools'))}", 1)
            if isinstance(input_content, str):
                self.log.debug_info(f"Input length: {len(input_content)} chars", 1)
                self.log.debug_info("Input content (first 500 chars):", 1)
                # Show more of the input - split by lines for readability
                input_preview = input_content[:500]
                for line in input_preview.split('\n'):
                    self.log.debug_info(f"{line}", 2)
                if len(input_content) > 500:
                    self.log.debug_info(f"... ({len(input_content) - 500} more chars)", 2)
            else:
                self.log.debug_info(f"Input: {len(input_content)} messages", 1)
                
            # Make API call
            response = self.client.responses.create(**params)
            
            self.log.debug_info("LLM Response:")
            if hasattr(response, 'output_text'):
                self.log.debug_info(f"Output length: {len(response.output_text)} chars", 1)
                self.log.debug_info("Output content (first 800 chars):", 1)
                # Show more of the output
                output_preview = response.output_text[:800]
                for line in output_preview.split('\n'):
                    self.log.debug_info(f"{line}", 2)
                if len(response.output_text) > 800:
                    self.log.debug_info(f"... ({len(response.output_text) - 800} more chars)", 2)

            return response
            
        except Exception as e:
            self.log.error(f"LLM call failed: {e}")
            raise
    

    def _extract_algorithm_from_text(
        self, source: Union[str, "openai.types.Response"]
    ) -> Dict[str, Any]:
        """Extract the algorithm code from the response."""

        # Normalise to plain text
        if hasattr(source, "output_text"):            # Response object
            text = source.output_text or ""
            # Get format info, handling Pydantic models
            text_obj = getattr(source, "text", None)
            if text_obj and hasattr(text_obj, "format"):
                fmt_obj = text_obj.format
                # Convert Pydantic model to dict if needed
                if hasattr(fmt_obj, "model_dump"):
                    fmt = fmt_obj.model_dump()
                elif hasattr(fmt_obj, "dict"):
                    fmt = fmt_obj.dict()
                else:
                    fmt = {}
            else:
                fmt = {}
        else:                                         # Raw string
            text, fmt = str(source), {}

        # Code-block extraction (3-pass strategy)
        code_block: Optional[str] = None

        # a) Structured `format:"code"` mode – easiest
        if isinstance(fmt, dict) and fmt.get("type") == "code":   
            code_block = text.strip()

        # b) Fenced ```python … ``` first (also accept ```py)
        if code_block is None:
            m = re.search(r"```(?:python|py)?\s*([\s\S]+?)```", text, re.IGNORECASE)
            if m:
                code_block = m.group(1).strip()
                
        # b2) Check for JSON mode response {"code": "..."}
        if code_block is None:
            try:
                # Try to parse as JSON first
                json_data = json.loads(text)
                if isinstance(json_data, dict) and 'code' in json_data:
                    code_block = json_data['code'].strip()
            except json.JSONDecodeError:
                pass
                
        # b3) Generic triple-backtick without language tag (for cases where model uses json fence)
        if code_block is None:
            m = re.search(r"```\s*([\s\S]+?)```", text)
            if m:
                code_block = m.group(1).strip()

        # c) Fallback – first top-level def
        if code_block is None:
            func_pat = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
            m = func_pat.search(text)
            if m:
                start = m.start()
                indent = len(m.group(0)) - len(m.group(0).lstrip())
                end_pat = re.compile(rf"^[^\S\r\n]{{0,{indent}}}(?:def|class)\s", re.MULTILINE)
                end = end_pat.search(text, m.end())
                code_block = text[start : end.start() if end else None].rstrip()

        if code_block is None:
            raise ValueError("Could not locate algorithm code in response")

        # Guardrail: ensure function name is correct
        if "def regularize_llm(" not in code_block:
            m = re.search(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", code_block, re.MULTILINE)
            if m:
                found_name = m.group(1)
                # If there is exactly one top-level function and it's not the expected name, fail fast
                funcs = re.findall(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", code_block, re.MULTILINE)
                if len(funcs) == 1:
                    raise ValueError(f"Expected function 'regularize_llm', found '{found_name}'")
        
        # Assemble & return
        return {"code": code_block}


    def _extract_json_from_text(
        self, source: Union[str, "openai.types.Response"]
    ) -> Dict[str, Any]:
        """
        Extract the first valid JSON object from either a Response or plain text.
        Prefers the structured JSON-mode output when available.
        """

        # 1) Structured JSON mode (`text.format.type == "json_object"`)
        if hasattr(source, "output_text"):
            text_obj = getattr(source, "text", None)
            if text_obj and hasattr(text_obj, "format"):
                fmt_obj = text_obj.format
                # Check if it's JSON mode
                if hasattr(fmt_obj, "type") and fmt_obj.type == "json_object":
                    try:
                        return json.loads(source.output_text)
                    except json.JSONDecodeError as e:
                        self.log.warning(f"JSON mode response failed to parse: {e}")
                elif hasattr(fmt_obj, "model_dump"):
                    fmt_dict = fmt_obj.model_dump()
                    if fmt_dict.get("type") == "json_object":
                        try:
                            return json.loads(source.output_text)
                        except json.JSONDecodeError as e:
                            self.log.warning(f"JSON mode response failed to parse: {e}")

        text = source.output_text if hasattr(source, "output_text") else str(source)
        
        # Truncate extremely long responses to prevent parsing issues
        if len(text) > 50000:  # 50KB limit
            self.log.warning(f"Response too long ({len(text)} chars), truncating to first 50KB")
            text = text[:50000]

        # 2) ```json …``` fence
        fence = re.search(r"```json\s*([\s\S]+?)```", text, re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass

        # 3) Balanced-brace sweep (properly handles braces inside quotes)
        depth, start = 0, None
        in_string = False
        escape_next = False
        
        for i, ch in enumerate(text):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                continue
                
            if ch == '\\' and in_string:
                escape_next = True
                continue
                
            # Toggle string state
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
                
            # Only count braces outside of strings
            if not in_string:
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}' and depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        snippet = text[start : i + 1]
                        try:
                            return json.loads(snippet)
                        except json.JSONDecodeError:
                            start = None

        # 4) Try to extract a reasonable JSON object from the beginning
        # Look for the first complete JSON object in the first 10KB
        search_text = text[:10000]
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(search_text):
            if char == '{' and start_idx is None:
                start_idx = i
                brace_count = 1
            elif char == '{' and start_idx is not None:
                brace_count += 1
            elif char == '}' and start_idx is not None:
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_str = search_text[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        start_idx = None
                        brace_count = 0

        # 5) Give up – return error info
        self.log.error(f"Failed to extract JSON from response of length {len(text)}")
        return {
            "error": "Failed to parse JSON response",
            "response_length": len(text),
            "response_preview": text[:500] + "..." if len(text) > 500 else text
        }
    
    def web_search_context(self, prompt: str, user_context: str) -> Dict[str, Any]:
        """Perform web search on experiment setup and ptychography.
        
        Uses the web search API.
        """
        self.log.llm(f"Starting web search with model: {self.reasoning_model}")
        
        # web_search.md
        search_prompt = prompt.format(
            user_context=user_context
        )
        
        # Fallback to regular model with web search
        response = self._call_llm(
            input_content=search_prompt,
            force_web_search=True,  # Always use web search for research
            use_reasoning=True
        )
        
        search_results = {
            'results': response.output_text,
            'timestamp': time.time()
        }
        if self.debug:
            self.log.debug_info(f"[DEBUG] Web search results:")
            self.log.debug_info(f"Results: {search_results['results']}", 1)
            self.log.debug_info(f"Timestamp: {search_results['timestamp']}", 1)
        
        return search_results


    def generate_algorithm(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new regularization algorithm with web search."""
        self.log.llm(f"Generating algorithm with {self.reasoning_model}")
        if self.web_search_enabled:
            self.log.llm("Web search enabled")
        
        # Format the prompt (discovery.md) with context
        formatted_prompt = prompt.format(
            recent_algorithms=json.dumps(context.get('recent_algorithms', []), indent=2),
            best_performance=json.dumps(context.get('best_performance', {}), indent=2),
            experiment_context=context.get('experiment_context', ''),
            aggregated_suggestions=json.dumps(context.get('aggregated_suggestions', []), indent=2),
        )
        
        response = self._call_llm(
            input_content=formatted_prompt,
            tools=[{"type": "web_search_preview"}] if self.web_search_enabled else None,
            use_reasoning=True
        )
        
        # Extract algorithm from response
        result = self._extract_algorithm_from_text(response)
        
        if self.debug and 'code' in result:
            self.log.debug_info("[DEBUG] Extracted algorithm code:")
            self.log.debug_info(f"Code length: {len(result['code'])} chars", 1)
            self.log.debug_info("Code content (first 15 lines):", 1)
            # Show more lines of code
            code_lines = result['code'].split('\n')[:15]
            for line in code_lines:
                self.log.debug_info(f"{line}", 2)
            if len(result['code'].split('\n')) > 15:
                remaining_lines = len(result['code'].split('\n')) - 15
                self.log.debug_info(f"... ({remaining_lines} more lines)", 2)
        
        return result
    
    def tune_parameters(self, prompt: str, algorithm: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to tune parameters in algorithm code."""
        self.log.llm(f"Tuning parameters for algorithm {algorithm.get('id', 'unknown')}")
        
        # Extract parameters from analysis if not directly available
        parameters = algorithm.get('parameters', {})
        if not parameters and 'analysis' in algorithm:
            parameters = algorithm['analysis'].get('parameters', {})
        
        # parameter_tuning.md
        tuning_prompt = prompt.format(
            code=algorithm['code'],
            current_metrics=json.dumps(algorithm.get('metrics', {}), indent=2),
            current_analysis=json.dumps(algorithm.get('analysis', {}), indent=2),
            parameters=json.dumps(parameters, indent=2),
        )
        
        # Use reasoning model for better parameter optimization
        response = self._call_llm(
            input_content=tuning_prompt,
            use_reasoning=True,
            disable_web_search=True  # No need for web search when tuning parameters
        )
        
        # Extract tuned algorithm
        tuned_algorithm = self._extract_algorithm_from_text(response)

        return tuned_algorithm
    
    def crossover_algorithms(self, prompt: str, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform crossover between pairs of algorithms using LLM."""
        evolved = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                # Crossover two algorithms
                parent1, parent2 = population[i], population[i + 1]
                # Use crossover-specific template
                crossover_prompt = prompt.format(
                    parent1_code=parent1['code'],
                    parent1_metrics=json.dumps(parent1['metrics'], indent=2),
                    parent1_analysis=json.dumps(parent1.get('analysis', {}), indent=2),
                    parent2_code=parent2['code'],
                    parent2_metrics=json.dumps(parent2['metrics'], indent=2),
                    parent2_analysis=json.dumps(parent2.get('analysis', {}), indent=2)
                )
                response = self._call_llm(
                    input_content=crossover_prompt,
                    use_reasoning=True,  # Add reasoning for better crossover decisions
                    disable_web_search=True  # No need for web search in crossover
                )
                algorithm = self._extract_algorithm_from_text(response)
                algorithm['parents'] = [parent1['id'], parent2['id']]
                algorithm['actual_strategy'] = 'crossover'  # Mark actual strategy
                evolved.append(algorithm)
            elif i < len(population):
                # Handle odd population size - mutate the last algorithm instead of dropping it
                parent = population[i]
                mutation_prompt = self.prompts['evolution_mutation'].format(
                    parent_code=parent['code'],
                    parent_metrics=json.dumps(parent['metrics'], indent=2),
                    parent_analysis=json.dumps(parent.get('analysis', {}), indent=2)
                )
                response = self._call_llm(
                    input_content=mutation_prompt,
                    use_reasoning=True,  # Add reasoning for better mutations
                    disable_web_search=True  # No need for web search in mutation
                )
                algorithm = self._extract_algorithm_from_text(response)
                algorithm['parent_id'] = parent['id']
                algorithm['actual_strategy'] = 'mutation'  # Mark actual strategy
                evolved.append(algorithm)
        return evolved

    def mutate_algorithms(self, prompt: str, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mutate each algorithm in the population using LLM."""
        evolved = []
        for parent in population:
            # Use mutation-specific template
            mutation_prompt = prompt.format(
                parent_code=parent['code'],
                parent_metrics=json.dumps(parent['metrics'], indent=2),
                parent_analysis=json.dumps(parent.get('analysis', {}), indent=2)
            )
            response = self._call_llm(
                input_content=mutation_prompt,
                use_reasoning=True,  # Add reasoning for better mutations
                disable_web_search=True  # No need for web search in mutation
            )
            algorithm = self._extract_algorithm_from_text(response)
            algorithm['parent_id'] = parent['id']
            evolved.append(algorithm)
        return evolved

    def evolve_algorithms(self, prompt: str, population: List[Dict[str, Any]], 
                         strategy: str) -> List[Dict[str, Any]]:
        """Evolve algorithms using LLM instead of genetic operators."""
        if strategy == 'crossover':
            return self.crossover_algorithms(prompt, population)
        elif strategy == 'mutation':
            return self.mutate_algorithms(prompt, population)
        else:
            raise ValueError(f"Unknown evolution strategy: {strategy}")

    def correct_algorithm(self, prompt: str, algorithm: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Correct algorithm based on error feedback."""
        self.log.llm(f"Correcting algorithm error: {error[:50]}...")
        
        # algorithm_correction.md
        correction_prompt = prompt.format(
            code=algorithm['code'],
            error=error
        )
        
        response = self._call_llm(
            input_content=correction_prompt,
            use_reasoning=False,
            disable_web_search=True  # No need for web search when correcting errors
        )
        
        # Extract corrected algorithm
        return self._extract_algorithm_from_text(response)


    def analyze_results(self, prompt: str, algorithm: Dict[str, Any], 
                       history: DiscoveryHistory) -> Dict[str, Any]:
        """Analyze algorithm performance and extract insights."""
        self.log.llm(f"Analyzing results with reasoning model: {self.reasoning_model}")
        
        # analysis.md
        analysis_prompt = prompt.format(
            code=algorithm['code'],
            metrics=json.dumps(algorithm.get('metrics', {}), indent=2),
            error=algorithm.get('error', 'None'),
            history_summary=json.dumps(history.get_context(), indent=2)
        )
        
        # Use reasoning for better analysis
        response = self._call_llm(
            input_content=analysis_prompt,
            use_reasoning=True,
            json_mode=True,
            disable_web_search=True  # Analysis of existing results doesn't need web search
        )
        
        # Extract JSON from response
        analysis = self._extract_json_from_text(response)
        
        # Add raw response text for debugging if JSON extraction failed or returned an error
        if 'raw_text' in analysis or 'error' in analysis:
            analysis['_debug_raw_response'] = response.output_text if hasattr(response, 'output_text') else str(response)
         
        if self.debug:
            self.log.debug_info("[DEBUG] Analysis results:")
            self.log.debug_info(f"Success: {algorithm.get('success', False)}", 1)
            if 'techniques' in analysis:
                self.log.debug_info(f"Techniques found ({len(analysis['techniques'])}):", 1)
                for tech in analysis['techniques'][:5]:
                    self.log.debug_info(f"- {tech}", 2)
            if 'parameters' in analysis:
                self.log.debug_info(f"Parameters extracted: {len(analysis['parameters'])}", 1)
                for param, value in list(analysis['parameters'].items())[:3]:
                    self.log.debug_info(f"- {param}: {value}", 2)
            if 'suggestions' in analysis:
                self.log.debug_info("Suggestions:", 1)
                for sug in analysis.get('suggestions', [])[:3]:
                    self.log.debug_info(f"- {sug}", 2)
            
        return analysis

   
    def analyze_code_security(self, code: str, context: str = "") -> Dict[str, Any]:
        """Analyze code for security risks before execution.
        
        Args:
            code: The Python code to analyze
            context: Additional context about the code's purpose
            
        Returns:
            Dict with keys:
                - is_safe: bool
                - issues: List[str]
        """
        security_prompt = """Analyze this Python code for security risks in a scientific computing context.

Context: {context}

Code to analyze:
```python
{code}
```

This code will be executed using exec() in a ptychography reconstruction algorithm.
The code should only perform mathematical operations on numpy/torch tensors.

Check for:
1. System calls (os.system, subprocess, etc.)
2. File I/O operations (open, Path operations, etc.)
3. Network access (requests, urllib, etc.)
4. Import of dangerous modules
5. Access to __builtins__ or attempts to break sandbox
6. Infinite loops or excessive resource usage
7. Any attempt to access data outside the provided tensors

Respond with JSON:
{{
    "is_safe": boolean,
    "issues": ["list of specific security issues found"],
}}"""

        formatted_prompt = security_prompt.format(
            context=context,
            code=code
        )
        
        try:
            response = self._call_llm(
                input_content=formatted_prompt,
                json_mode=True,
                disable_web_search=True  # Static code analysis doesn't need web search
            )
            
            result = self._extract_json_from_text(response)
            
            return result
            
        except Exception as e:
            self.log.error(f"Security analysis failed: {e}")
            # Default to unsafe on error
            return {
                "is_safe": False,
                "issues": [f"Security analysis failed: {str(e)}"],
            }
 
