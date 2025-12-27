#!/usr/bin/env python3
"""
Structured Output Reliability Test Suite
Tests JSON generation reliability across open-source LLMs

Measures:
1. Parse failures (invalid JSON)
2. Schema violations (missing required fields, wrong types)
3. Hallucinated fields (fields not in schema)
4. Silent failures (valid JSON but semantically wrong)
"""

import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List
from enum import Enum
import statistics

logger = logging.getLogger("StructuredOutputTest")


def load_knowledge_base(kb_path: str) -> List[dict]:
    """Load knowledge base rules from JSON file."""
    if not os.path.exists(kb_path):
        logger.warning(f"Knowledge base not found: {kb_path}")
        return []

    try:
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
        logger.info(f"Loaded {len(kb_data)} KB rules from {kb_path}")
        return kb_data
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        return []


def build_kb_guidance_text(kb_rules: List[dict]) -> str:
    """Build guidance text from KB rules to inject into prompt."""
    if not kb_rules:
        return ""

    # Filter for JSON-relevant rules
    json_rules = [r for r in kb_rules if r.get("domain") == "json"]
    if not json_rules:
        return ""

    guidance_parts = ["IMPORTANT RULES FOR JSON OUTPUT:"]
    for rule in json_rules:
        guidance_parts.append(f"- {rule.get('text', '')}")

    return "\n".join(guidance_parts)


# Schema definitions - increasingly complex nested structures
SCHEMAS = {
    "simple": {
        "name": "SimpleOrder",
        "description": "A simple e-commerce order",
        "schema": {
            "type": "object",
            "required": ["order_id", "customer_name", "total"],
            "properties": {
                "order_id": {"type": "string"},
                "customer_name": {"type": "string"},
                "total": {"type": "number"},
                "status": {"type": "string", "enum": ["pending", "shipped", "delivered"]}
            },
            "additionalProperties": False
        }
    },
    "medium": {
        "name": "UserProfile",
        "description": "A user profile with nested address and preferences",
        "schema": {
            "type": "object",
            "required": ["user_id", "email", "address", "preferences"],
            "properties": {
                "user_id": {"type": "integer"},
                "email": {"type": "string", "format": "email"},
                "address": {
                    "type": "object",
                    "required": ["street", "city", "country", "postal_code"],
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                        "postal_code": {"type": "string"}
                    },
                    "additionalProperties": False
                },
                "preferences": {
                    "type": "object",
                    "required": ["newsletter", "theme"],
                    "properties": {
                        "newsletter": {"type": "boolean"},
                        "theme": {"type": "string", "enum": ["light", "dark", "system"]},
                        "language": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    },
    "complex": {
        "name": "APIResponse",
        "description": "A complex API response with nested arrays and objects",
        "schema": {
            "type": "object",
            "required": ["request_id", "timestamp", "data", "pagination", "metadata"],
            "properties": {
                "request_id": {"type": "string", "pattern": "^[a-f0-9-]{36}$"},
                "timestamp": {"type": "string", "format": "date-time"},
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "type", "attributes"],
                        "properties": {
                            "id": {"type": "integer"},
                            "type": {"type": "string", "enum": ["user", "product", "order"]},
                            "attributes": {
                                "type": "object",
                                "required": ["name", "created_at"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "created_at": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "additionalProperties": False
                            },
                            "relationships": {
                                "type": "object",
                                "properties": {
                                    "parent_id": {"type": ["integer", "null"]},
                                    "children_ids": {
                                        "type": "array",
                                        "items": {"type": "integer"}
                                    }
                                },
                                "additionalProperties": False
                            }
                        },
                        "additionalProperties": False
                    }
                },
                "pagination": {
                    "type": "object",
                    "required": ["page", "per_page", "total", "total_pages"],
                    "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                        "total": {"type": "integer", "minimum": 0},
                        "total_pages": {"type": "integer", "minimum": 0}
                    },
                    "additionalProperties": False
                },
                "metadata": {
                    "type": "object",
                    "required": ["version", "rate_limit"],
                    "properties": {
                        "version": {"type": "string"},
                        "rate_limit": {
                            "type": "object",
                            "required": ["remaining", "reset_at"],
                            "properties": {
                                "remaining": {"type": "integer"},
                                "reset_at": {"type": "string"}
                            },
                            "additionalProperties": False
                        },
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    },
    "edge_case": {
        "name": "FinancialTransaction",
        "description": "Edge cases: nullable fields, specific formats, constraints",
        "schema": {
            "type": "object",
            "required": ["transaction_id", "amount", "currency", "parties", "status"],
            "properties": {
                "transaction_id": {"type": "string", "minLength": 10, "maxLength": 20},
                "amount": {"type": "number", "minimum": 0, "exclusiveMinimum": True},
                "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"]},
                "exchange_rate": {"type": ["number", "null"]},
                "parties": {
                    "type": "object",
                    "required": ["sender", "receiver"],
                    "properties": {
                        "sender": {
                            "type": "object",
                            "required": ["account_id", "name"],
                            "properties": {
                                "account_id": {"type": "string"},
                                "name": {"type": "string"},
                                "bank_code": {"type": ["string", "null"]}
                            },
                            "additionalProperties": False
                        },
                        "receiver": {
                            "type": "object",
                            "required": ["account_id", "name"],
                            "properties": {
                                "account_id": {"type": "string"},
                                "name": {"type": "string"},
                                "bank_code": {"type": ["string", "null"]}
                            },
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False
                },
                "status": {"type": "string", "enum": ["pending", "processing", "completed", "failed", "reversed"]},
                "fees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "amount"],
                        "properties": {
                            "type": {"type": "string"},
                            "amount": {"type": "number", "minimum": 0}
                        },
                        "additionalProperties": False
                    }
                },
                "notes": {"type": ["string", "null"], "maxLength": 500}
            },
            "additionalProperties": False
        }
    }
}

# Test prompts for each schema complexity level
TEST_PROMPTS = {
    "simple": [
        "Generate a JSON object for an order with ID 'ORD-12345' for customer John Smith, total $99.99, status pending.",
        "Create order JSON: order_id='ORD-99999', customer Sarah Jones, $250.00 total, delivered status.",
        "Output a simple order object in JSON format for order ABC123, customer 'Test User', amount 50, shipped."
    ],
    "medium": [
        "Generate a user profile JSON with user_id 42, email john@example.com, address at 123 Main St, New York, USA, 10001, newsletter enabled, dark theme.",
        "Create a JSON user profile: ID 100, alice@test.org, living at 456 Oak Ave, London, UK, SW1A 1AA, no newsletter, light theme, English language.",
        "Output user profile JSON for user 7, email test@demo.com, address 789 Pine Rd, Toronto, Canada, M5V 2T6, newsletter true, system theme."
    ],
    "complex": [
        """Generate a complex API response JSON with:
- request_id: a valid UUID like 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
- timestamp: '2024-01-15T10:30:00Z'
- data array with 2 items of type 'user', each with id, attributes (name, created_at, tags array)
- pagination: page 1, 10 per page, 25 total, 3 total pages
- metadata with version '2.0', rate_limit (remaining: 99, reset_at timestamp)""",
        
        """Create an API response JSON:
- request_id: UUID format 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
- current timestamp in ISO format
- data: 3 products with sequential IDs, names, created dates, and category tags
- pagination for page 2 of results (5 per page, 12 total items)
- metadata v1.5, 45 rate limit remaining""",
    ],
    "edge_case": [
        """Generate a financial transaction JSON:
- transaction_id: 'TXN-1234567890' (10-20 chars)
- amount: 1500.50 (must be > 0)
- currency: USD
- exchange_rate: null (not applicable)
- sender: account 'ACC001', name 'Alice Corp', bank_code 'CHASE001'
- receiver: account 'ACC002', name 'Bob Inc', bank_code null
- status: completed
- fees: [{type: 'processing', amount: 2.50}, {type: 'wire', amount: 15.00}]
- notes: 'Monthly payment'""",
        
        """Create transaction JSON with edge cases:
- transaction_id: exactly 15 characters
- amount: 0.01 (minimum valid positive amount)
- currency: EUR
- exchange_rate: 1.08
- parties with minimal info (no optional bank codes)
- status: pending
- empty fees array
- notes: null"""
    ]
}


@dataclass
class ValidationError:
    error_type: str  # parse_failure, missing_field, wrong_type, invalid_enum, hallucinated_field, constraint_violation
    field_path: str
    expected: str
    actual: str
    severity: str  # critical, major, minor


@dataclass 
class TestResult:
    model: str
    schema_name: str
    prompt_index: int
    raw_response: str
    parsed_json: Optional[dict]
    is_valid_json: bool
    errors: list = field(default_factory=list)
    hallucinated_fields: list = field(default_factory=list)
    latency_ms: float = 0.0


def extract_json_from_response(response: str) -> tuple[Optional[dict], Optional[str]]:
    """Try multiple strategies to extract JSON from model response."""
    
    # Strategy 1: Direct parse (response is pure JSON)
    try:
        return json.loads(response.strip()), None
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON in code blocks
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip()), None
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Find JSON object boundaries
    # Find outermost { } pair
    start = response.find('{')
    if start != -1:
        depth = 0
        for i, char in enumerate(response[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[start:i+1]), None
                    except json.JSONDecodeError as e:
                        return None, f"Found JSON-like structure but failed to parse: {e}"
    
    return None, "No JSON structure found in response"


def get_all_fields(schema: dict, prefix: str = "") -> set:
    """Recursively get all valid field names from a schema."""
    fields = set()
    
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        for name, prop_schema in props.items():
            full_path = f"{prefix}.{name}" if prefix else name
            fields.add(full_path)
            fields.update(get_all_fields(prop_schema, full_path))
    elif schema.get("type") == "array":
        items = schema.get("items", {})
        fields.update(get_all_fields(items, f"{prefix}[]"))
    
    return fields


def find_hallucinated_fields(data: Any, schema: dict, path: str = "") -> list[str]:
    """Find fields in data that don't exist in schema."""
    hallucinated = []
    
    if isinstance(data, dict):
        schema_props = schema.get("properties", {})
        allows_additional = schema.get("additionalProperties", True)
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in schema_props:
                if not allows_additional:
                    hallucinated.append(current_path)
            else:
                # Recurse into nested structures
                hallucinated.extend(
                    find_hallucinated_fields(value, schema_props[key], current_path)
                )
    
    elif isinstance(data, list):
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            hallucinated.extend(
                find_hallucinated_fields(item, items_schema, f"{path}[{i}]")
            )
    
    return hallucinated


def validate_against_schema(data: Any, schema: dict, path: str = "") -> list[ValidationError]:
    """Validate data against JSON schema and return all errors."""
    errors = []
    
    schema_type = schema.get("type")
    
    # Handle union types like ["string", "null"]
    if isinstance(schema_type, list):
        if data is None and "null" in schema_type:
            return errors
        # Try to validate against non-null type
        for t in schema_type:
            if t != "null":
                schema_type = t
                break
    
    # Type checking
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None)
    }
    
    if schema_type and schema_type != "null":
        expected_type = type_map.get(schema_type)
        if expected_type and not isinstance(data, expected_type):
            # Special case: integer can be satisfied by int (not bool)
            if schema_type == "integer" and isinstance(data, bool):
                errors.append(ValidationError(
                    "wrong_type", path, schema_type, type(data).__name__, "major"
                ))
            elif not isinstance(data, expected_type):
                errors.append(ValidationError(
                    "wrong_type", path, schema_type, type(data).__name__, "major"
                ))
                return errors  # Can't validate further if type is wrong
    
    # Object validation
    if schema_type == "object" and isinstance(data, dict):
        # Check required fields
        required = schema.get("required", [])
        for req_field in required:
            if req_field not in data:
                errors.append(ValidationError(
                    "missing_field", f"{path}.{req_field}" if path else req_field,
                    "required field", "missing", "critical"
                ))
        
        # Validate each property
        props = schema.get("properties", {})
        for key, value in data.items():
            if key in props:
                errors.extend(validate_against_schema(
                    value, props[key], f"{path}.{key}" if path else key
                ))
    
    # Array validation
    if schema_type == "array" and isinstance(data, list):
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            errors.extend(validate_against_schema(
                item, items_schema, f"{path}[{i}]"
            ))
    
    # Enum validation
    if "enum" in schema and data not in schema["enum"]:
        errors.append(ValidationError(
            "invalid_enum", path, str(schema["enum"]), str(data), "major"
        ))
    
    # String constraints
    if schema_type == "string" and isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(ValidationError(
                "constraint_violation", path, 
                f"minLength {schema['minLength']}", f"length {len(data)}", "minor"
            ))
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(ValidationError(
                "constraint_violation", path,
                f"maxLength {schema['maxLength']}", f"length {len(data)}", "minor"
            ))
        if "pattern" in schema:
            if not re.match(schema["pattern"], data):
                errors.append(ValidationError(
                    "constraint_violation", path,
                    f"pattern {schema['pattern']}", data, "minor"
                ))
    
    # Number constraints
    if schema_type in ("number", "integer") and isinstance(data, (int, float)):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(ValidationError(
                "constraint_violation", path,
                f"minimum {schema['minimum']}", str(data), "minor"
            ))
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(ValidationError(
                "constraint_violation", path,
                f"maximum {schema['maximum']}", str(data), "minor"
            ))
        if schema.get("exclusiveMinimum") and data <= schema.get("minimum", float('-inf')):
            errors.append(ValidationError(
                "constraint_violation", path,
                f"exclusive minimum {schema.get('minimum')}", str(data), "minor"
            ))
    
    return errors


def run_single_test(model_func, model_name: str, schema_name: str,
                    prompt_index: int, prompt: str, schema: dict,
                    verbose: bool = False, kb_guidance: str = "") -> TestResult:
    """Run a single test and collect results."""

    # Build prompt with optional KB guidance
    kb_section = f"\n{kb_guidance}\n" if kb_guidance else ""

    full_prompt = f"""You must respond with ONLY valid JSON that conforms to this schema. No other text, no explanations, no markdown formatting - just the raw JSON object.
{kb_section}
Schema:
{json.dumps(schema, indent=2)}

Task: {prompt}

Respond with ONLY the JSON object:"""

    if verbose:
        print(f"\n{'─'*50}")
        print(f"[PROMPT] {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    start_time = time.time()
    try:
        response = model_func(full_prompt)
    except Exception as e:
        response = f"ERROR: {e}"
        if verbose:
            print(f"[ERROR] Model call failed: {e}")
    latency_ms = (time.time() - start_time) * 1000

    if verbose:
        print(f"[LATENCY] {latency_ms:.0f}ms")
        print(f"[RAW RESPONSE] {response[:300]}{'...' if len(response) > 300 else ''}")

    result = TestResult(
        model=model_name,
        schema_name=schema_name,
        prompt_index=prompt_index,
        raw_response=response,
        parsed_json=None,
        is_valid_json=False,
        latency_ms=latency_ms
    )

    # Try to parse JSON
    parsed, parse_error = extract_json_from_response(response)

    if parsed is None:
        result.errors.append(ValidationError(
            "parse_failure", "", "valid JSON", parse_error or "parse failed", "critical"
        ))
        if verbose:
            print(f"[PARSE] ❌ FAILED - {parse_error}")
        return result

    result.parsed_json = parsed
    result.is_valid_json = True

    if verbose:
        print(f"[PARSE] ✅ Valid JSON extracted")
        print(f"[PARSED JSON] {json.dumps(parsed, indent=2)[:500]}{'...' if len(json.dumps(parsed)) > 500 else ''}")

    # Validate against schema
    schema_errors = validate_against_schema(parsed, schema)
    result.errors.extend(schema_errors)

    if verbose and schema_errors:
        print(f"[VALIDATION] ⚠️  {len(schema_errors)} schema errors:")
        for err in schema_errors:
            print(f"    - [{err.severity}] {err.error_type}: {err.field_path} (expected {err.expected}, got {err.actual})")
    elif verbose:
        print(f"[VALIDATION] ✅ Schema valid")

    # Find hallucinated fields
    result.hallucinated_fields = find_hallucinated_fields(parsed, schema)
    for field in result.hallucinated_fields:
        result.errors.append(ValidationError(
            "hallucinated_field", field, "not in schema", "present", "major"
        ))

    if verbose and result.hallucinated_fields:
        print(f"[HALLUCINATION] ⚠️  Found {len(result.hallucinated_fields)} extra fields: {result.hallucinated_fields}")
    elif verbose:
        print(f"[HALLUCINATION] ✅ No extra fields")

    return result


class ModelInterface:
    """Interface for testing different models."""

    @staticmethod
    def ollama(model_name: str):
        """Create Ollama model function."""
        import requests

        def call(prompt: str) -> str:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for deterministic output
                        "num_predict": 2048
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]

        return call

    @staticmethod
    def transformers_from_config(config: dict):
        """Create HuggingFace transformers model function from config dict."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_path = config["model_path"]
        model_name = config.get("display_name", config.get("name", model_path))
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(config.get("dtype", "bfloat16"), torch.bfloat16)
        use_flash = config.get("use_flash_attention", False)
        max_memory = config.get("max_memory_gb")

        print(f"\n[LOADING] {model_name} from {model_path}")
        print(f"  dtype: {config.get('dtype', 'bfloat16')}, flash_attention: {use_flash}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_kwargs = {
            "dtype": dtype,  # Updated from dtype (deprecated)
            "device_map": "auto",
        }

        # Try flash attention if requested, fall back gracefully if not available
        if use_flash:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"  ✅ Flash Attention 2 enabled")
            except ImportError:
                print(f"  ⚠️  Flash Attention 2 not installed, using default attention")

        if max_memory:
            model_kwargs["max_memory"] = {0: f"{max_memory}GB"}

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        print(f"  ✅ Model loaded on {model.device}")

        def call(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            return response[len(prompt):].strip()

        return call

    @staticmethod
    def transformers(model_name: str, device: str = "auto"):
        """Create HuggingFace transformers model function (legacy)."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device
        )

        def call(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            return response[len(prompt):].strip()

        return call


def load_model_configs(config_path: str = "models.json") -> list[dict]:
    """Load model configs from JSON file. Only returns enabled models (configs only, not loaded)."""
    config_file = os.path.join(os.path.dirname(__file__), config_path)

    if not os.path.exists(config_file):
        print(f"Warning: {config_path} not found")
        return []

    with open(config_file, "r") as f:
        config = json.load(f)

    enabled_configs = []
    for model_config in config.get("models", []):
        if model_config.get("enabled", False):
            enabled_configs.append(model_config)

    return enabled_configs


def unload_model(model, tokenizer):
    """Unload model and clear GPU/MPS memory."""
    import gc
    import torch

    print("\n[UNLOADING] Clearing model from memory...")

    # Move model to CPU first before deleting (helps with MPS)
    try:
        model.to("cpu")
    except Exception:
        pass

    # Delete model and tokenizer
    del model
    del tokenizer

    # Clear Python garbage
    gc.collect()

    # Clear PyTorch cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  ✅ CUDA cache cleared")
        elif torch.backends.mps.is_available():
            # MPS memory management
            torch.mps.synchronize()
            torch.mps.empty_cache()
            print("  ✅ MPS cache cleared")
    except Exception as e:
        print(f"  ⚠️  Cache clear warning: {e}")

    # Extra gc passes
    gc.collect()
    gc.collect()
    print("  ✅ Memory cleared")


def run_test_suite_sequential(model_configs: list[dict], iterations: int = 3, verbose: bool = False, use_kb: bool = False) -> list[TestResult]:
    """Run test suite loading one model at a time, unloading between models."""
    from model_runner import ModelRunner

    all_results = []
    total_models = len(model_configs)
    total_tests_per_model = sum(len(TEST_PROMPTS.get(s, [])) for s in SCHEMAS) * iterations
    total_tests = total_tests_per_model * total_models
    current_test = 0

    # Load knowledge base if requested
    kb_guidance = ""
    if use_kb:
        kb_path = os.path.join(os.path.dirname(__file__), "config", "knowledge_json.json")
        kb_rules = load_knowledge_base(kb_path)
        if kb_rules:
            kb_guidance = build_kb_guidance_text(kb_rules)
            print(f"\n  📚 Knowledge base loaded: {len(kb_rules)} rules")
            if verbose:
                print(f"  KB guidance:\n{kb_guidance[:300]}...")
        else:
            print(f"\n  ⚠️  No knowledge base found at: {kb_path}")

    for model_idx, config in enumerate(model_configs):
        model_name = config.get("display_name", config.get("name"))
        model_path = config["model_path"]

        print(f"\n{'='*60}")
        print(f"[{model_idx + 1}/{total_models}] Loading model: {model_name}")
        print('='*60)

        try:
            runner = ModelRunner(model_path)
            runner.load()
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            continue

        # Create callable wrapper for this model
        def make_model_call(r):
            def call(prompt: str) -> str:
                response, _ = r.generate(prompt, max_new_tokens=2048)
                return response
            return call

        model_func = make_model_call(runner)

        # Run tests for this model
        print(f"\n  Starting tests for {model_name}...")
        if kb_guidance:
            print(f"  📚 Using KB prompt guidance")

        for schema_name, schema_def in SCHEMAS.items():
            print(f"\n  Schema: {schema_name} ({schema_def.get('description', '')})")
            prompts = TEST_PROMPTS.get(schema_name, [])

            for prompt_idx, prompt in enumerate(prompts):
                for iteration in range(iterations):
                    current_test += 1
                    progress = f"[{current_test}/{total_tests}]"

                    if not verbose:
                        print(f"    {progress} Prompt {prompt_idx + 1}, iteration {iteration + 1}...", end=" ")

                    result = run_single_test(
                        model_func, model_name, schema_name,
                        prompt_idx, prompt, schema_def["schema"],
                        verbose=verbose, kb_guidance=kb_guidance
                    )
                    all_results.append(result)

                    if not verbose:
                        if not result.is_valid_json:
                            print(f"❌ PARSE FAIL ({result.latency_ms:.0f}ms)")
                        elif result.errors:
                            print(f"⚠️  {len(result.errors)} errors ({result.latency_ms:.0f}ms)")
                        else:
                            print(f"✅ OK ({result.latency_ms:.0f}ms)")
                    else:
                        status = "✅ PASS" if not result.errors else f"❌ FAIL ({len(result.errors)} errors)"
                        print(f"[RESULT] {progress} {status} - {result.latency_ms:.0f}ms")

        # Unload model before loading next one
        runner.unload()

    return all_results


def generate_report(results: list[TestResult]) -> dict:
    """Generate comprehensive analysis report."""
    
    report = {
        "summary": {},
        "by_model": {},
        "by_schema": {},
        "failure_analysis": {
            "parse_failures": [],
            "schema_violations": [],
            "hallucinated_fields": [],
            "type_errors": [],
            "constraint_violations": []
        },
        "smoking_guns": []
    }
    
    # Group by model
    models = set(r.model for r in results)
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        
        parse_failures = sum(1 for r in model_results if not r.is_valid_json)
        schema_valid = sum(1 for r in model_results if r.is_valid_json and not r.errors)
        has_hallucinations = sum(1 for r in model_results if r.hallucinated_fields)
        
        report["by_model"][model] = {
            "total_tests": total,
            "parse_success_rate": (total - parse_failures) / total * 100,
            "full_schema_compliance": schema_valid / total * 100,
            "hallucination_rate": has_hallucinations / total * 100,
            "avg_errors_per_response": statistics.mean(len(r.errors) for r in model_results),
            "avg_latency_ms": statistics.mean(r.latency_ms for r in model_results),
            "error_breakdown": {}
        }
        
        # Error type breakdown
        error_types = {}
        for r in model_results:
            for e in r.errors:
                error_types[e.error_type] = error_types.get(e.error_type, 0) + 1
        report["by_model"][model]["error_breakdown"] = error_types
    
    # Group by schema complexity
    for schema_name in SCHEMAS:
        schema_results = [r for r in results if r.schema_name == schema_name]
        if not schema_results:
            continue
        
        total = len(schema_results)
        report["by_schema"][schema_name] = {
            "total_tests": total,
            "parse_success_rate": sum(1 for r in schema_results if r.is_valid_json) / total * 100,
            "full_compliance_rate": sum(1 for r in schema_results if r.is_valid_json and not r.errors) / total * 100
        }
    
    # Find smoking guns - examples of silent failures
    for r in results:
        if r.is_valid_json and r.hallucinated_fields:
            report["smoking_guns"].append({
                "model": r.model,
                "schema": r.schema_name,
                "description": "Valid JSON but invented fields not in schema",
                "hallucinated_fields": r.hallucinated_fields,
                "sample_json": r.parsed_json
            })
        
        # Valid JSON, correct structure, but wrong types
        type_errors = [e for e in r.errors if e.error_type == "wrong_type"]
        if r.is_valid_json and type_errors:
            report["smoking_guns"].append({
                "model": r.model,
                "schema": r.schema_name,
                "description": "Valid JSON but wrong field types",
                "type_errors": [(e.field_path, e.expected, e.actual) for e in type_errors]
            })
    
    # Overall summary
    total = len(results)
    report["summary"] = {
        "total_tests": total,
        "models_tested": list(models),
        "schemas_tested": list(SCHEMAS.keys()),
        "overall_parse_success": sum(1 for r in results if r.is_valid_json) / total * 100,
        "overall_full_compliance": sum(1 for r in results if r.is_valid_json and not r.errors) / total * 100,
        "total_smoking_guns": len(report["smoking_guns"])
    }
    
    return report


def print_report(report: dict):
    """Pretty print the analysis report."""
    
    print("\n" + "="*80)
    print("STRUCTURED OUTPUT RELIABILITY REPORT")
    print("="*80)
    
    print("\n📊 OVERALL SUMMARY")
    print("-"*40)
    s = report["summary"]
    print(f"  Total tests: {s['total_tests']}")
    print(f"  Models tested: {', '.join(s['models_tested'])}")
    print(f"  Overall parse success: {s['overall_parse_success']:.1f}%")
    print(f"  Overall full schema compliance: {s['overall_full_compliance']:.1f}%")
    print(f"  Smoking gun failures found: {s['total_smoking_guns']}")
    
    print("\n📈 BY MODEL")
    print("-"*40)
    for model, stats in report["by_model"].items():
        print(f"\n  {model}:")
        print(f"    Parse success rate: {stats['parse_success_rate']:.1f}%")
        print(f"    Full schema compliance: {stats['full_schema_compliance']:.1f}%")
        print(f"    Hallucination rate: {stats['hallucination_rate']:.1f}%")
        print(f"    Avg errors per response: {stats['avg_errors_per_response']:.2f}")
        print(f"    Avg latency: {stats['avg_latency_ms']:.0f}ms")
        if stats["error_breakdown"]:
            print(f"    Error breakdown: {stats['error_breakdown']}")
    
    print("\n📉 BY SCHEMA COMPLEXITY")
    print("-"*40)
    for schema, stats in report["by_schema"].items():
        print(f"  {schema}: {stats['full_compliance_rate']:.1f}% compliance ({stats['total_tests']} tests)")
    
    if report["smoking_guns"]:
        print("\n🚨 SMOKING GUN FAILURES (Silent Failures)")
        print("-"*40)
        for i, sg in enumerate(report["smoking_guns"][:10], 1):  # Limit to 10
            print(f"\n  {i}. [{sg['model']}] {sg['description']}")
            if "hallucinated_fields" in sg:
                print(f"     Hallucinated: {sg['hallucinated_fields']}")
            if "type_errors" in sg:
                for path, expected, actual in sg["type_errors"]:
                    print(f"     {path}: expected {expected}, got {actual}")


def main():
    """Main entry point - configure and run tests."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Structured Output Reliability Test Suite")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output showing prompts, responses, and detailed validation")
    parser.add_argument("-i", "--iterations", type=int, default=2,
                        help="Number of iterations per prompt (default: 2)")
    parser.add_argument("--models", type=str, default="models.json",
                        help="Path to models config file (default: models.json)")
    parser.add_argument("--kb", action="store_true",
                        help="Enable knowledge base prompt guidance")
    parser.add_argument("--no-kb", action="store_true",
                        help="Disable knowledge base (default)")
    args = parser.parse_args()

    print("="*80)
    print("STRUCTURED OUTPUT RELIABILITY TEST SUITE")
    print("="*80)
    print("\nThis test suite measures JSON generation reliability across LLMs.")
    print("It detects parse failures, schema violations, and hallucinated fields.")
    print("\n⚡ Sequential mode: Load one model, test, unload, then next model")

    if args.verbose:
        print("🔍 VERBOSE MODE ENABLED - showing detailed output for each test")

    # Load model configs (not the models themselves)
    model_configs = load_model_configs(args.models)

    if not model_configs:
        print("\n❌ No models enabled in models.json")
        print("Please enable at least one model by setting 'enabled': true")
        return None

    model_names = [c.get("display_name", c.get("name")) for c in model_configs]
    print(f"\nModels to test: {model_names}")
    print(f"Schemas to test: {list(SCHEMAS.keys())}")
    print(f"Prompts per schema: {[len(TEST_PROMPTS[s]) for s in SCHEMAS]}")
    print(f"Iterations per prompt: {args.iterations}")
    total_tests = sum(len(TEST_PROMPTS.get(s, [])) for s in SCHEMAS) * len(model_configs) * args.iterations
    print(f"Total tests to run: {total_tests}")
    print(f"Knowledge base: {'ENABLED' if args.kb else 'DISABLED'}")

    # Run tests sequentially (load one model at a time)
    print("\nStarting tests...")
    results = run_test_suite_sequential(
        model_configs,
        iterations=args.iterations,
        verbose=args.verbose,
        use_kb=args.kb
    )

    if not results:
        print("\n❌ No test results - all models may have failed to load")
        return None

    # Generate and print report
    report = generate_report(results)
    print_report(report)

    # Save detailed results with timestamp and model names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_names_str = "_".join(
        name.lower().replace(" ", "-").replace(".", "")[:20]
        for name in model_names
    )

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "models_tested": model_names,
        "report": report,
        "raw_results": [
            {
                "model": r.model,
                "schema": r.schema_name,
                "prompt_index": r.prompt_index,
                "is_valid_json": r.is_valid_json,
                "parsed_json": r.parsed_json,
                "errors": [
                    {"type": e.error_type, "path": e.field_path,
                     "expected": e.expected, "actual": e.actual}
                    for e in r.errors
                ],
                "hallucinated_fields": r.hallucinated_fields,
                "latency_ms": r.latency_ms,
                "raw_response": r.raw_response[:500]  # Truncate
            }
            for r in results
        ]
    }

    # Save with timestamp and model names
    filename = f"results_{model_names_str}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Also save a "latest" symlink/copy for convenience
    latest_path = os.path.join(results_dir, f"latest_{timestamp}.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nDetailed results saved to: {filepath}")
    print(f"Latest results also at: {latest_path}")

    return report


if __name__ == "__main__":
    main()
