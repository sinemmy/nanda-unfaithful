"""Model loading and generation utilities for Qwen-14B."""

import torch
import logging
from typing import Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and managing the Qwen-14B model."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load the Qwen-14B model and tokenizer from HuggingFace."""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Using device: {self.config.device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading configuration
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
        }
        
        # Use device_map for automatic device placement if CUDA
        if self.config.device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
        
        # Load model
        logger.info("Loading model weights (this may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Set to eval mode and disable gradients
        self.model.eval()
        torch.set_grad_enabled(False)
        
        logger.info("Model loaded successfully!")
        self.log_memory_usage()
        
        return self.model, self.tokenizer
    
    def log_memory_usage(self):
        """Log GPU memory usage if available."""
        if self.config.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """Generate response with explicit CoT reasoning.
        
        Returns:
            Tuple of (cot_reasoning, final_answer)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Use config defaults if not specified
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Format prompt with CoT instructions
        cot_prompt = self._format_cot_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            cot_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with sampling
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode response
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse CoT and final answer
        cot_reasoning, final_answer = self._parse_response(generated)
        
        return cot_reasoning, final_answer
    
    def _format_cot_prompt(self, prompt: str) -> str:
        """Format prompt to encourage chain-of-thought reasoning."""
        cot_instruction = """
Let me think through this step by step.

<thinking>
[Your detailed reasoning here]
</thinking>

Based on my analysis, here is my response:
"""
        return f"{prompt}\n\n{cot_instruction}"
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse response to extract CoT reasoning and final answer."""
        # Try parsing with BeautifulSoup first for more robust handling
        try:
            soup = BeautifulSoup(response, "lxml")
            
            # Look for thinking tags
            thinking_tag = soup.find("thinking")
            if thinking_tag:
                cot_reasoning = thinking_tag.get_text().strip()
                
                # Remove the thinking tag from soup
                thinking_tag.decompose()
                
                # Final answer is remaining text
                final_answer = soup.get_text().strip()
            else:
                # Also check for answer tags (like Anthropic's classifiers)
                answer_tag = soup.find("answer")
                if answer_tag:
                    final_answer = answer_tag.get_text().strip()
                    # Everything before answer tag is reasoning
                    answer_tag.decompose()
                    cot_reasoning = soup.get_text().strip()
                else:
                    # No structured tags found
                    cot_reasoning = response
                    final_answer = response
                    
        except Exception as e:
            logger.debug(f"BeautifulSoup parsing failed, falling back to string parsing: {e}")
            
            # Fallback to simple string parsing
            if "<thinking>" in response and "</thinking>" in response:
                start = response.index("<thinking>") + len("<thinking>")
                end = response.index("</thinking>")
                cot_reasoning = response[start:end].strip()
                
                # Final answer is everything after </thinking>
                final_answer = response[end + len("</thinking>"):].strip()
            else:
                # No tags found at all
                cot_reasoning = response
                final_answer = response
        
        # Clean up common prefixes from final answer
        prefixes = [
            "Based on my analysis, here is my response:",
            "Based on my analysis,",
            "My response:",
            "Here is my response:",
            "Here's my response:",
        ]
        for prefix in prefixes:
            if final_answer.startswith(prefix):
                final_answer = final_answer[len(prefix):].strip()
                break
        
        return cot_reasoning, final_answer
    
    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("Model unloaded and GPU cache cleared")