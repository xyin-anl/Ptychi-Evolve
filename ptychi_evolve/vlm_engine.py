"""Vision Language Model (VLM) engine for OpenAI vision API integration."""

import base64
import json
from pathlib import Path
import re
import io
from typing import Any, Dict, List, Tuple, Union
import openai
from openai import OpenAI
from PIL import Image
from ptychi_evolve.logging import get_logger



class VLMEngine:
    """VLM engine for evaluation using OpenAI's vision API."""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False, debug: bool = False):
        """Initialize VLM engine.
        """
        self.config = config
        # Prefer evaluation.vlm, fallback to top-level vlm
        self.vlm_config = (config.get('evaluation', {}).get('vlm')
                           or config.get('vlm', {})
                           or {})
        self.model = self.vlm_config.get('model', 'gpt-4.1')
        self.verbose = verbose or config.get('verbose', False)
        self.debug = debug or config.get('debug', False)
        
        # Initialize logger
        self.log = get_logger(__name__, verbose=self.verbose, debug=self.debug)
        
        self.client = OpenAI()
        self.few_shot_examples = []
        
        self.log.info(f"VLM Engine initialized with model: {self.model}")
        
    def encode_image(self, image_path: Path) -> Tuple[str, str]:
        """Encode image to base64 string for API input.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_type, base64_encoded_string)
        """
        # Get file suffix without the dot
        suffix = image_path.suffix.lower().lstrip('.')
        
        if suffix in ['png', 'jpg', 'jpeg']:
            with open(image_path, "rb") as image_file:
                return suffix, base64.b64encode(image_file.read()).decode('utf-8')
        elif suffix in ['tiff', 'tif']:
            # Convert TIFF to PNG and encode
            with Image.open(image_path) as img:
                # Check if it's a multi-page TIFF
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    self.log.warning(f"Multi-page TIFF detected ({img.n_frames} pages). Only the first page will be used.")
                
                # Convert to RGB if necessary (some TIFFs might be in different modes)
                if img.mode not in ('RGB', 'RGBA', 'L'):
                    img = img.convert('RGB')
                
                # Save to bytes buffer as PNG
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Return as PNG with base64 encoding
                return 'png', base64.b64encode(buffer.read()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {suffix}. Supported types: png, jpg, jpeg, tiff, tif")
        
        
    
    def add_few_shot_example(self, image_path: Path, evaluation: Any):
        """Add a few-shot example for learning.
        
        Args:
            image_path: Path to the example image
            evaluation: Either a dictionary with metrics or a natural language description string
        """
        image_type, image_base64 = self.encode_image(image_path)
        example = {
            'image_type': image_type,
            'image_base64': image_base64,
            'image_path': str(image_path),
            'example_evaluation': evaluation
        }
        self.few_shot_examples.append(example)
        self.log.info(f"Added few-shot example from {image_path}. Total examples: {len(self.few_shot_examples)}")

    def _build_few_shot_messages(self, image_path: Path) -> List[Dict]:
        """Build messages for few-shot learning."""
        messages = [
            {
                "role": "developer",
                "content": "You are an expert in evaluating ptychographic phase reconstruction quality. "
                          "You will be shown examples of reconstructions with their structured evaluations, "
                          "then asked to evaluate a new reconstruction following the same structured format."
            }
        ]
        
        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples):
            eval_data = example['example_evaluation']
            base64_image = example['image_base64']
            image_type = example['image_type']
            # Add the example image
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Example {i+1}: Please evaluate this ptychographic reconstruction."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
            # Add the structured evaluation
            messages.append({
                "role": "assistant",
                "content": json.dumps(eval_data, indent=2)
            })
        
        # Add the new image to evaluate
        image_type, base64_image = self.encode_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Now evaluate this new reconstruction, please provide a structured evaluation in JSON format following the same structure as the examples"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        })
        
        return messages

    def _extract_json_from_text(
        self, source: Union[str, "openai.types.Response"]
    ) -> Dict[str, Any]:
        """Extract JSON from response text."""
        text = source.output_text if hasattr(source, "output_text") else str(source)
        
        # Try JSON fence first
        fence = re.search(r"```json\s*([\s\S]+?)```", text, re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Return raw text as fallback
        return {"raw_text": text}

    def evaluate_with_few_shot(self, image_path: Path) -> Dict[str, Any]:
        """Evaluate reconstruction using few-shot learning.
        
        Args:
            image_path: Path to the reconstruction image

        Returns:
            Dictionary containing VLM evaluation results
        """
        if not self.few_shot_examples:
            raise ValueError("No few-shot examples provided. Please add examples first.")
        
        # Build messages for few-shot learning
        messages = self._build_few_shot_messages(image_path)
        
        # Call OpenAI Responses API with vision support
        try:
            # Convert messages format to Responses API input format
            response = self.client.responses.create(
                model=self.model,
                input=messages,
            )
            
            return self._extract_json_from_text(response)
            
        except Exception as e:
            self.log.error(f"VLM API call failed: {e}")
            raise

        
    def _build_description_messages(self, image_path: Path, evaluation_description: str) -> List[Dict]:
        """Build messages for description-based evaluation."""
        image_type, base64_image = self.encode_image(image_path)
        
        messages = [
            {
                "role": "developer",
                "content": "You are an expert in evaluating ptychographic phase reconstruction quality. "
                          "You will be shown an image of a ptychographic reconstruction, "
                          "then asked to evaluate the reconstruction based on the provided criteria. "
                          f"Evaluation Criteria: {evaluation_description}"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Please evaluate this ptychographic reconstruction.

Please provide your evaluation in JSON format:
{{
    "quality_score": 0.0-1.0,
    "feedback": "Detailed explanation of the quality of the reconstruction according to the criteria",
}}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        return messages


    def evaluate_with_description(self, image_path: Path, evaluation_description: str) -> Dict[str, Any]:
        """Evaluate reconstruction using natural language description.
        
        Args:
            image_path: Path to the reconstruction image
            evaluation_description: Natural language description of evaluation criteria
            
        Returns:
            Dictionary containing VLM evaluation results
        """
        # Build messages for description-based evaluation
        messages = self._build_description_messages(
            image_path, evaluation_description
        )
        
        # Call OpenAI Responses API with vision support
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
            )
            
            return self._extract_json_from_text(response)
            
        except Exception as e:
            self.log.error(f"VLM API call failed: {e}")
            raise
            
