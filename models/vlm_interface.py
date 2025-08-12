"""
Vision-Language Model Interface for Object Counting

This module provides a unified interface for interacting with different
Vision-Language Models (VLMs) to perform object counting tasks.
"""

import os
import json
import base64
import time
import re
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. GPT-4V will not work.")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Local models will not work.")


class VLMInterface(ABC):
    """Abstract base class for Vision-Language Model interfaces."""
    
    @abstractmethod
    def count_objects(self, image_base64: str, object_type: str, **kwargs) -> Dict[str, Any]:
        """Count objects in an image.
        
        Args:
            image_base64: Base64 encoded image
            object_type: Type of object to count (e.g., 'person', 'car')
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with count, confidence, and other metadata
        """
        pass
    
    def extract_number_from_text(self, text: str) -> int:
        """Extract a number from text response using various strategies."""
        if not text:
            return 0
        
        text_lower = text.lower().strip()
        
        # Strategy 1: Look for explicit numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            # Prefer smaller numbers (more likely to be counts)
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num <= 100:  # Reasonable range for object counting
                    return num
            # Fallback to first number
            return int(numbers[0])
        
        # Strategy 2: Look for written numbers
        number_words = {
            'zero': 0, 'none': 0, 'no': 0,
            'one': 1, 'single': 1, 'a': 1,
            'two': 2, 'couple': 2, 'pair': 2,
            'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        for word, num in number_words.items():
            if word in text_lower:
                return num
        
        # Strategy 3: Look for quantity indicators
        if any(word in text_lower for word in ['many', 'multiple', 'several', 'various']):
            return 3  # Reasonable guess for "many"
        
        if any(word in text_lower for word in ['few', 'some']):
            return 2  # Reasonable guess for "few"
        
        # Default fallback
        return 0


class GPT4VInterface(VLMInterface):
    """Interface for OpenAI GPT-4 with Vision."""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required for GPT-4V. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.max_retries = max_retries
        
    def count_objects(self, image_base64: str, object_type: str, **kwargs) -> Dict[str, Any]:
        """Count objects using GPT-4V."""
        
        # Enhanced prompt for better counting performance
        prompt = self._create_counting_prompt(object_type, **kwargs)
        
        for attempt in range(self.max_retries):
            try:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1,  # Low temperature for consistent counting
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                
                return {
                    'count': int(result.get('count', 0)),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', ''),
                    'visible_objects': result.get('visible_objects', []),
                    'raw_response': result_text,
                    'model': 'gpt-4o'
                }
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    # Try to extract count from raw text
                    count = self.extract_number_from_text(response.choices[0].message.content)
                    return {
                        'count': count,
                        'confidence': 0.3,
                        'reasoning': f'JSON parsing failed, extracted count from text: {response.choices[0].message.content}',
                        'error': f'JSON decode error: {str(e)}',
                        'model': 'gpt-4o'
                    }
                    
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        'count': 0,
                        'confidence': 0.0,
                        'error': str(e),
                        'reasoning': f'Error after {self.max_retries} attempts',
                        'model': 'gpt-4o'
                    }
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return {
            'count': 0,
            'confidence': 0.0,
            'error': 'Max retries exceeded',
            'model': 'gpt-4o'
        }
    
    def _create_counting_prompt(self, object_type: str, **kwargs) -> str:
        """Create an optimized prompt for object counting."""
        
        difficulty_context = kwargs.get('difficulty', 'unknown')
        scenario_type = kwargs.get('scenario_type', 'general')
        
        base_prompt = f"""Look at this image very carefully and count the number of {object_type} that you can see.

IMPORTANT INSTRUCTIONS:
- Count ONLY objects that are clearly identifiable as {object_type}
- Include partially visible {object_type} if you can clearly identify them
- Include {object_type} that might be partially hidden or occluded by other objects
- Be especially careful to look for {object_type} that might blend into the background
- Do not count the same {object_type} twice
- If you're unsure about whether something is a {object_type}, do not count it"""
        
        if scenario_type == 'camouflage':
            base_prompt += f"""
- Pay extra attention to {object_type} that might be camouflaged or blend with their surroundings
- Look for subtle differences in texture, shape, or pattern that might indicate a {object_type}
- Consider that some {object_type} might be deliberately hidden or naturally camouflaged"""
        
        if difficulty_context in ['hard', 'extreme']:
            base_prompt += f"""
- This is a challenging image - look very carefully
- Consider lighting, shadows, and perspective that might make {object_type} hard to spot
- Double-check areas where {object_type} might be partially obscured"""
        
        base_prompt += f"""

Please respond with ONLY a JSON object in this exact format:
{{
    "count": <number>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<detailed explanation of what you see and your counting process>",
    "visible_objects": ["<brief description of each {object_type} you counted>"]
}}

Be precise in your counting and provide an honest confidence level."""
        
        return base_prompt


class BLIP2Interface(VLMInterface):
    """Interface for BLIP-2 via HuggingFace Inference API."""
    
    def __init__(self, hf_token: Optional[str] = None, max_retries: int = 3):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.max_retries = max_retries
        self.api_url = "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b"
        
    def count_objects(self, image_base64: str, object_type: str, **kwargs) -> Dict[str, Any]:
        """Count objects using BLIP-2."""
        
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Create question
        question = self._create_counting_question(object_type, **kwargs)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=image_bytes,
                    params={"question": question},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        answer = result[0].get('answer', '')
                    elif isinstance(result, dict):
                        answer = result.get('answer', '')
                    else:
                        answer = str(result)
                    
                    # Extract count from answer
                    count = self.extract_number_from_text(answer)
                    
                    # Estimate confidence based on answer characteristics
                    confidence = self._estimate_confidence(answer, count)
                    
                    return {
                        'count': count,
                        'confidence': confidence,
                        'reasoning': answer,
                        'raw_response': str(result),
                        'model': 'blip2-opt-2.7b'
                    }
                    
                elif response.status_code == 503:
                    # Model loading, retry with exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Model loading, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    if attempt == self.max_retries - 1:
                        return {
                            'count': 0,
                            'confidence': 0.0,
                            'error': f'HTTP {response.status_code}: {response.text}',
                            'reasoning': 'API request failed',
                            'model': 'blip2-opt-2.7b'
                        }
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        'count': 0,
                        'confidence': 0.0,
                        'error': str(e),
                        'reasoning': f'Error after {self.max_retries} attempts',
                        'model': 'blip2-opt-2.7b'
                    }
                
                time.sleep(2 ** attempt)
        
        return {
            'count': 0,
            'confidence': 0.0,
            'error': 'Max retries exceeded',
            'model': 'blip2-opt-2.7b'
        }
    
    def _create_counting_question(self, object_type: str, **kwargs) -> str:
        """Create question for BLIP-2 counting."""
        
        scenario_type = kwargs.get('scenario_type', 'general')
        
        if scenario_type == 'camouflage':
            return f"How many {object_type} are in this image? Look carefully for ones that might be hidden or camouflaged."
        else:
            return f"How many {object_type} can you see in this image? Count all visible ones including partially hidden ones."
    
    def _estimate_confidence(self, answer: str, count: int) -> float:
        """Estimate confidence based on answer characteristics."""
        if not answer:
            return 0.1
        
        answer_lower = answer.lower()
        
        # High confidence indicators
        if any(phrase in answer_lower for phrase in ['clearly', 'exactly', 'precisely', 'definitely']):
            base_confidence = 0.9
        elif any(str(i) in answer for i in range(20)):  # Contains specific number
            base_confidence = 0.8
        else:
            base_confidence = 0.6
        
        # Reduce confidence for uncertainty indicators
        if any(phrase in answer_lower for phrase in ['maybe', 'might', 'possibly', 'unsure', 'difficult']):
            base_confidence *= 0.7
        
        # Reduce confidence for very high counts (likely errors)
        if count > 20:
            base_confidence *= 0.5
        
        return min(1.0, max(0.1, base_confidence))


class LLaVAInterface(VLMInterface):
    """Interface for LLaVA (Local inference - requires GPU)."""
    
    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", max_retries: int = 3):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for LLaVA. Install with: pip install transformers torch")
        
        self.model_path = model_path
        self.max_retries = max_retries
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Lazy load the model when needed."""
        if self.model is None:
            try:
                logger.info(f"Loading LLaVA model: {self.model_path}")
                # Note: This is a placeholder - actual LLaVA loading would require specific setup
                # For now, we'll use HuggingFace Inference API as a fallback
                self.api_url = f"https://api-inference.huggingface.co/models/{self.model_path}"
                logger.info("Using HuggingFace Inference API for LLaVA")
            except Exception as e:
                logger.error(f"Failed to load LLaVA model: {e}")
                raise
    
    def count_objects(self, image_base64: str, object_type: str, **kwargs) -> Dict[str, Any]:
        """Count objects using LLaVA."""
        
        self._load_model()
        
        # Use HuggingFace Inference API as fallback
        return self._count_via_api(image_base64, object_type, **kwargs)
    
    def _count_via_api(self, image_base64: str, object_type: str, **kwargs) -> Dict[str, Any]:
        """Count using HuggingFace Inference API."""
        
        hf_token = os.getenv("HF_TOKEN")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        image_bytes = base64.b64decode(image_base64)
        prompt = f"How many {object_type} are in this image? Count carefully and provide only the number."
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=image_bytes,
                    params={"prompt": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        answer = result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        answer = result.get('generated_text', result.get('answer', ''))
                    else:
                        answer = str(result)
                    
                    count = self.extract_number_from_text(answer)
                    confidence = 0.7 if count > 0 else 0.3
                    
                    return {
                        'count': count,
                        'confidence': confidence,
                        'reasoning': answer,
                        'raw_response': str(result),
                        'model': 'llava-1.5-7b'
                    }
                    
                elif response.status_code == 503:
                    wait_time = 2 ** attempt
                    logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    if attempt == self.max_retries - 1:
                        return {
                            'count': 0,
                            'confidence': 0.0,
                            'error': f'HTTP {response.status_code}: {response.text}',
                            'model': 'llava-1.5-7b'
                        }
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        'count': 0,
                        'confidence': 0.0,
                        'error': str(e),
                        'model': 'llava-1.5-7b'
                    }
                time.sleep(2 ** attempt)
        
        return {
            'count': 0,
            'confidence': 0.0,
            'error': 'Max retries exceeded',
            'model': 'llava-1.5-7b'
        }


class VLMManager:
    """Main interface for managing multiple VLMs."""
    
    def __init__(self, openai_key: Optional[str] = None, hf_token: Optional[str] = None, 
                 confidence_threshold: float = 0.5, max_retries: int = 3):
        """Initialize VLM manager with API keys and configuration."""
        
        self.openai_key = openai_key
        self.hf_token = hf_token
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        
        # Initialize available models
        self.models = {}
        
        # GPT-4V
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.models['GPT-4V'] = GPT4VInterface(openai_key, max_retries)
                logger.info("GPT-4V initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GPT-4V: {e}")
        
        # BLIP-2
        try:
            self.models['BLIP-2'] = BLIP2Interface(hf_token, max_retries)
            logger.info("BLIP-2 initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize BLIP-2: {e}")
        
        # LLaVA
        try:
            self.models['LLaVA'] = LLaVAInterface(max_retries=max_retries)
            logger.info("LLaVA initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLaVA: {e}")
        
        logger.info(f"Initialized {len(self.models)} VLM models: {list(self.models.keys())}")
    
    def count_objects(self, model_name: str, image_base64: str, object_type: str, **kwargs) -> Dict[str, Any]:
        """Count objects using specified model."""
        
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
        
        try:
            result = self.models[model_name].count_objects(image_base64, object_type, **kwargs)
            
            # Add metadata
            result['model_name'] = model_name
            result['object_type'] = object_type
            result['timestamp'] = time.time()
            
            # Flag low confidence results
            if result.get('confidence', 0) < self.confidence_threshold:
                result['low_confidence'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            return {
                'count': 0,
                'confidence': 0.0,
                'error': str(e),
                'model_name': model_name,
                'object_type': object_type,
                'timestamp': time.time()
            }
    
    def count_objects_multi_model(self, model_names: List[str], image_base64: str, 
                                  object_type: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Count objects using multiple models."""
        
        results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                logger.info(f"Running {model_name} for {object_type} counting...")
                results[model_name] = self.count_objects(model_name, image_base64, object_type, **kwargs)
            else:
                logger.warning(f"Model {model_name} not available, skipping...")
                results[model_name] = {
                    'count': 0,
                    'confidence': 0.0,
                    'error': f'Model {model_name} not available',
                    'model_name': model_name,
                    'object_type': object_type,
                    'timestamp': time.time()
                }
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        info = {}
        
        for model_name, model_instance in self.models.items():
            info[model_name] = {
                'class': model_instance.__class__.__name__,
                'available': True,
                'max_retries': getattr(model_instance, 'max_retries', 3)
            }
            
            if hasattr(model_instance, 'api_url'):
                info[model_name]['api_url'] = model_instance.api_url
            if hasattr(model_instance, 'model_path'):
                info[model_name]['model_path'] = model_instance.model_path
        
        return info


# Convenience function for easy usage
def create_vlm_interface(openai_key: Optional[str] = None, hf_token: Optional[str] = None, **kwargs) -> VLMManager:
    """Create a VLM interface with default settings."""
    return VLMManager(
        openai_key=openai_key or os.getenv("OPENAI_API_KEY"),
        hf_token=hf_token or os.getenv("HF_TOKEN"),
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    vlm = create_vlm_interface()
    
    print("Available models:", vlm.get_available_models())
    print("Model info:", vlm.get_model_info())
    
    # Note: This would require an actual base64 encoded image
    # result = vlm.count_objects("GPT-4V", image_base64, "person")
    # print("Counting result:", result)
