"""
Reusable Model Runner for HuggingFace Transformers on MPS/CUDA/CPU.

Usage:
    from model_runner import ModelRunner

    runner = ModelRunner("/path/to/model")
    runner.load()
    response, latency = runner.generate("Your prompt here")
    runner.unload()

Or as context manager:
    with ModelRunner("/path/to/model") as runner:
        response, latency = runner.generate("Your prompt here")
"""

import time
import gc
from typing import Tuple, Optional, List
import torch


class ModelRunner:
    """Handles model loading and inference for HuggingFace models."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        """
        Initialize the model runner.

        Args:
            model_path: Path to HuggingFace model (local or hub)
            device: Device to use ("auto", "cuda", "mps", "cpu")
            dtype: Data type ("bfloat16", "float16", "float32")
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.device_preference = device
        self.dtype_str = dtype
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.device = None
        self.eos_token_ids = []

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False

    def load(self):
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from {self.model_path}...")
        start = time.time()

        # Determine device
        if self.device_preference == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                device_map = "auto"
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                device_map = "mps"
            else:
                self.device = torch.device("cpu")
                device_map = "cpu"
        else:
            self.device = torch.device(self.device_preference)
            device_map = self.device_preference

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype_str, torch.bfloat16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        # Build EOS token IDs for this model
        self._build_eos_tokens()

        elapsed = time.time() - start
        print(f"  Model loaded on {self.device} in {elapsed:.1f}s")
        print(f"  dtype: {self.dtype_str}, EOS tokens: {len(self.eos_token_ids)}")

    def _build_eos_tokens(self):
        """Build list of all EOS token IDs for this model."""
        self.eos_token_ids = []

        # Add tokenizer's EOS
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids.append(self.tokenizer.eos_token_id)

        # Add model's generation config EOS (may have multiple)
        if hasattr(self.model.generation_config, 'eos_token_id'):
            gen_eos = self.model.generation_config.eos_token_id
            if gen_eos is not None:
                if isinstance(gen_eos, list):
                    self.eos_token_ids.extend(gen_eos)
                else:
                    self.eos_token_ids.append(gen_eos)

        # Add model-specific stop tokens
        special_stops = [
            '<end_of_turn>',  # Gemma
            '<|eot_id|>',     # Llama 3
            '<|end_of_text|>',  # Llama 3
            '</s>',           # Mistral/Ministral
        ]
        for stop in special_stops:
            try:
                stop_id = self.tokenizer.convert_tokens_to_ids(stop)
                if stop_id is not None and stop_id != self.tokenizer.unk_token_id:
                    self.eos_token_ids.append(stop_id)
            except Exception:
                pass

        # Deduplicate
        self.eos_token_ids = list(set(self.eos_token_ids))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        use_chat_template: bool = True,
    ) -> Tuple[str, float]:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            use_chat_template: Whether to apply chat template

        Returns:
            Tuple of (response_text, latency_ms)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Format as chat if model expects it
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.device)

        # Generate
        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_ids if self.eos_token_ids else self.tokenizer.eos_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip(), latency_ms

    def unload(self):
        """Unload model and clear memory."""
        if self.model is None:
            return

        print(f"Unloading model...")

        # Move to CPU first (helps with MPS)
        try:
            self.model.to("cpu")
        except Exception:
            pass

        # Delete
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        # Clear memory
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()

        gc.collect()
        print(f"  Model unloaded, memory cleared")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


class MultiModelRunner:
    """Run inference across multiple models sequentially."""

    def __init__(self, model_paths: List[str], **kwargs):
        """
        Args:
            model_paths: List of model paths to test
            **kwargs: Arguments passed to ModelRunner
        """
        self.model_paths = model_paths
        self.runner_kwargs = kwargs

    def run_all(self, prompts: List[str], **generate_kwargs) -> dict:
        """
        Run prompts through all models.

        Args:
            prompts: List of prompts to run
            **generate_kwargs: Arguments passed to generate()

        Returns:
            Dict mapping model_path -> list of (response, latency) tuples
        """
        results = {}

        for model_path in self.model_paths:
            print(f"\n{'='*60}")
            print(f"Testing: {model_path}")
            print('='*60)

            runner = ModelRunner(model_path, **self.runner_kwargs)
            runner.load()

            model_results = []
            for i, prompt in enumerate(prompts):
                print(f"  Prompt {i+1}/{len(prompts)}...", end=" ")
                response, latency = runner.generate(prompt, **generate_kwargs)
                model_results.append((response, latency))
                print(f"{latency:.0f}ms")

            results[model_path] = model_results
            runner.unload()

        return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_runner.py <model_path> [prompt]")
        print("\nExample:")
        print("  python model_runner.py /path/to/model 'What is 2+2?'")
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What is 2+2? Answer briefly."

    with ModelRunner(model_path) as runner:
        response, latency = runner.generate(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        print(f"Latency: {latency:.0f}ms")
