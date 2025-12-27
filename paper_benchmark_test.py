#!/usr/bin/env python3
"""
Paper Benchmark Test Suite
Tests JSON generation reliability using test cases from research papers:

1. JSONSchemaBench (epfl-dlab) - Real-world schemas by difficulty
2. SchemaBench (thunlp) - Custom formats, escape handling, complex refs
3. StructuredRAG - Task complexity progression

Run with/without KB:
    python paper_benchmark_test.py                    # Baseline (no KB)
    python paper_benchmark_test.py --kb               # With KB guidance
    python paper_benchmark_test.py --kb -v            # Verbose with KB
"""

import json
import os
import re
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple
from datetime import datetime
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# KNOWLEDGE BASE SUPPORT
# ============================================================================

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


# ============================================================================
# TEST CASES FROM PAPERS
# ============================================================================

# SchemaBench examples (thunlp)
SCHEMABENCH_TESTS = [
    # Complex Schema task - with $ref
    {
        "source": "SchemaBench/complex_schema",
        "task": "complex_schema",
        "difficulty": "hard",
        "schema": {
            "type": "object",
            "required": ["user", "settings"],
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["name", "email"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "email": {"type": "string"},
                        "age": {"type": "integer", "minimum": 0}
                    },
                    "additionalProperties": False
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "theme": {"type": "string", "enum": ["light", "dark"]},
                        "notifications": {"type": "boolean"}
                    },
                    "required": ["theme", "notifications"],
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "prompt": "Generate a JSON object with a user (name: Alice, email: alice@example.com, age: 30) and settings (theme: dark, notifications: true)."
    },
    # Custom Formats task - minLength, phone format
    {
        "source": "SchemaBench/custom_formats",
        "task": "custom_formats",
        "difficulty": "medium",
        "schema": {
            "type": "object",
            "required": ["phone", "password", "file_path"],
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "US phone number"
                },
                "password": {
                    "type": "string",
                    "minLength": 8,
                    "description": "Password with at least 8 characters"
                },
                "file_path": {
                    "type": "string",
                    "description": "Linux file path starting with /"
                }
            },
            "additionalProperties": False
        },
        "prompt": "Generate JSON with phone: 555-123-4567, password: SecurePass123, file_path: /home/user/documents/file.txt"
    },
    # Escape Translation task
    {
        "source": "SchemaBench/escape_translation",
        "task": "escape_translation",
        "difficulty": "hard",
        "schema": {
            "type": "object",
            "required": ["message", "code_snippet"],
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A message containing quotes"
                },
                "code_snippet": {
                    "type": "string",
                    "description": "A code snippet with backslashes"
                }
            },
            "additionalProperties": False
        },
        "prompt": "Generate JSON with message: 'He said \"Hello World\"' and code_snippet: 'C:\\Users\\Admin\\file.txt'"
    },
    # Base64 format
    {
        "source": "SchemaBench/base64_format",
        "task": "base64_format",
        "difficulty": "medium",
        "schema": {
            "type": "object",
            "required": ["data", "encoding"],
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Base64 encoded string"
                },
                "encoding": {
                    "type": "string",
                    "enum": ["base64", "base64url"]
                }
            },
            "additionalProperties": False
        },
        "prompt": "Generate JSON with data as base64 encoded 'Hello World' (SGVsbG8gV29ybGQ=) and encoding: base64"
    }
]

# StructuredRAG tasks - complexity progression
STRUCTUREDRAG_TESTS = [
    # Task 1: String output (easiest)
    {
        "source": "StructuredRAG/string",
        "task": "string_output",
        "difficulty": "easy",
        "schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": {"type": "string"}
            },
            "additionalProperties": False
        },
        "prompt": "What is the capital of France? Return as JSON with 'answer' field only."
    },
    # Task 2: Integer output
    {
        "source": "StructuredRAG/integer",
        "task": "integer_output",
        "difficulty": "easy",
        "schema": {
            "type": "object",
            "required": ["count"],
            "properties": {
                "count": {"type": "integer"}
            },
            "additionalProperties": False
        },
        "prompt": "How many continents are there? Return as JSON with 'count' field as integer."
    },
    # Task 3: Boolean output
    {
        "source": "StructuredRAG/boolean",
        "task": "boolean_output",
        "difficulty": "easy",
        "schema": {
            "type": "object",
            "required": ["result"],
            "properties": {
                "result": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "prompt": "Is the Earth flat? Return as JSON with 'result' field as boolean (true/false)."
    },
    # Task 4: List of strings (harder)
    {
        "source": "StructuredRAG/list_strings",
        "task": "list_strings",
        "difficulty": "medium",
        "schema": {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "additionalProperties": False
        },
        "prompt": "List the first 5 planets from the sun. Return as JSON with 'items' array."
    },
    # Task 5: Composite object (AnswerWithConfidence)
    {
        "source": "StructuredRAG/composite",
        "task": "composite_object",
        "difficulty": "medium",
        "schema": {
            "type": "object",
            "required": ["answer", "confidence"],
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "additionalProperties": False
        },
        "prompt": "What year did World War 2 end? Return JSON with 'answer' (string) and 'confidence' (number 0-1)."
    },
    # Task 6: List of composite objects (hardest - massive failures in paper)
    {
        "source": "StructuredRAG/list_composite",
        "task": "list_composite",
        "difficulty": "hard",
        "schema": {
            "type": "object",
            "required": ["answers"],
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["answer", "confidence"],
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "additionalProperties": False
        },
        "prompt": "Name 3 programming languages with your confidence in each. Return JSON with 'answers' array of objects with 'answer' and 'confidence' fields."
    }
]

# JSONSchemaBench-style tests (increasing difficulty)
JSONSCHEMABENCH_TESTS = [
    # Easy - simple flat object
    {
        "source": "JSONSchemaBench/easy",
        "task": "simple_product",
        "difficulty": "easy",
        "schema": {
            "type": "object",
            "required": ["name", "price", "in_stock"],
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
                "in_stock": {"type": "boolean"}
            },
            "additionalProperties": False
        },
        "prompt": "Generate a product JSON: name 'Widget', price 29.99, in_stock true."
    },
    # Medium - nested object with enum
    {
        "source": "JSONSchemaBench/medium",
        "task": "order_with_shipping",
        "difficulty": "medium",
        "schema": {
            "type": "object",
            "required": ["order_id", "customer", "shipping"],
            "properties": {
                "order_id": {"type": "string"},
                "customer": {
                    "type": "object",
                    "required": ["name", "email"],
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "additionalProperties": False
                },
                "shipping": {
                    "type": "object",
                    "required": ["method", "address"],
                    "properties": {
                        "method": {"type": "string", "enum": ["standard", "express", "overnight"]},
                        "address": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "prompt": "Generate order JSON: order_id 'ORD-001', customer (name: John Doe, email: john@example.com), shipping (method: express, address: 123 Main St)."
    },
    # Hard - deep nesting with arrays
    {
        "source": "JSONSchemaBench/hard",
        "task": "api_response",
        "difficulty": "hard",
        "schema": {
            "type": "object",
            "required": ["status", "data", "meta"],
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "tags"],
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "additionalProperties": False
                    }
                },
                "meta": {
                    "type": "object",
                    "required": ["page", "total"],
                    "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "total": {"type": "integer", "minimum": 0}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "prompt": "Generate API response JSON: status 'success', data array with 2 items (id: 1, name: 'Item A', tags: ['new', 'featured']) and (id: 2, name: 'Item B', tags: ['sale']), meta (page: 1, total: 50)."
    },
    # Ultra - complex with nullable and constraints
    {
        "source": "JSONSchemaBench/ultra",
        "task": "financial_record",
        "difficulty": "ultra",
        "schema": {
            "type": "object",
            "required": ["transaction_id", "amount", "currency", "parties"],
            "properties": {
                "transaction_id": {"type": "string", "minLength": 10, "maxLength": 20},
                "amount": {"type": "number", "minimum": 0.01},
                "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                "exchange_rate": {"type": ["number", "null"]},
                "parties": {
                    "type": "object",
                    "required": ["sender", "receiver"],
                    "properties": {
                        "sender": {
                            "type": "object",
                            "required": ["id", "name"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "bank": {"type": ["string", "null"]}
                            },
                            "additionalProperties": False
                        },
                        "receiver": {
                            "type": "object",
                            "required": ["id", "name"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "bank": {"type": ["string", "null"]}
                            },
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False
                },
                "notes": {"type": ["string", "null"]}
            },
            "additionalProperties": False
        },
        "prompt": "Generate financial transaction JSON: transaction_id 'TXN-1234567890', amount 1500.50, currency USD, exchange_rate null, parties with sender (id: 'S001', name: 'Alice Corp', bank: 'Chase') and receiver (id: 'R001', name: 'Bob Inc', bank: null), notes: 'Monthly payment'."
    }
]

# Combine all test cases
ALL_TEST_CASES = SCHEMABENCH_TESTS + STRUCTUREDRAG_TESTS + JSONSCHEMABENCH_TESTS


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

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
    source: str
    task: str
    difficulty: str
    raw_response: str
    parsed_json: Optional[dict]
    is_valid_json: bool
    errors: list = field(default_factory=list)
    hallucinated_fields: list = field(default_factory=list)
    latency_ms: float = 0.0
    used_kb: bool = False


def extract_json_from_response(response: str) -> Tuple[Optional[dict], Optional[str]]:
    """Try multiple strategies to extract JSON from model response."""

    # Strategy 1: Direct parse
    try:
        return json.loads(response.strip()), None
    except json.JSONDecodeError:
        pass

    # Strategy 2: Code blocks
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip()), None
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find JSON boundaries
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
                        return None, f"JSON-like structure found but invalid: {e}"

    return None, "No JSON structure found"


def find_hallucinated_fields(data: Any, schema: dict, path: str = "") -> List[str]:
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


def validate_against_schema(data: Any, schema: dict, path: str = "") -> List[ValidationError]:
    """Validate data against JSON schema and return all errors."""
    errors = []

    schema_type = schema.get("type")

    # Handle union types like ["string", "null"]
    if isinstance(schema_type, list):
        if data is None and "null" in schema_type:
            return errors
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
            if schema_type == "integer" and isinstance(data, bool):
                errors.append(ValidationError(
                    "wrong_type", path or "root", schema_type, type(data).__name__, "major"
                ))
            elif not isinstance(data, expected_type):
                errors.append(ValidationError(
                    "wrong_type", path or "root", schema_type, type(data).__name__, "major"
                ))
                return errors

    # Object validation
    if schema_type == "object" and isinstance(data, dict):
        required = schema.get("required", [])
        for req_field in required:
            if req_field not in data:
                errors.append(ValidationError(
                    "missing_field", f"{path}.{req_field}" if path else req_field,
                    "required field", "missing", "critical"
                ))

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
            "invalid_enum", path or "root", str(schema["enum"]), str(data), "major"
        ))

    # String constraints
    if schema_type == "string" and isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(ValidationError(
                "constraint_violation", path or "root",
                f"minLength {schema['minLength']}", f"length {len(data)}", "minor"
            ))
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(ValidationError(
                "constraint_violation", path or "root",
                f"maxLength {schema['maxLength']}", f"length {len(data)}", "minor"
            ))

    # Number constraints
    if schema_type in ("number", "integer") and isinstance(data, (int, float)):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(ValidationError(
                "constraint_violation", path or "root",
                f"minimum {schema['minimum']}", str(data), "minor"
            ))
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(ValidationError(
                "constraint_violation", path or "root",
                f"maximum {schema['maximum']}", str(data), "minor"
            ))

    return errors


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_single_test(model_func, model_name: str, test_case: dict,
                    verbose: bool = False, kb_guidance: str = "") -> TestResult:
    """Run a single test case."""

    schema = test_case["schema"]
    prompt = test_case.get("prompt", "Generate valid JSON matching the schema.")

    # Build the full prompt
    if kb_guidance:
        full_prompt = f"""{kb_guidance}

You must respond with ONLY valid JSON that conforms to this schema.
No other text, no explanations, no markdown - just the raw JSON object.

Schema:
{json.dumps(schema, indent=2)}

Task: {prompt}

Respond with ONLY the JSON object:"""
    else:
        full_prompt = f"""You must respond with ONLY valid JSON that conforms to this schema.
No other text, no explanations, no markdown - just the raw JSON object.

Schema:
{json.dumps(schema, indent=2)}

Task: {prompt}

Respond with ONLY the JSON object:"""

    start_time = time.time()
    try:
        response = model_func(full_prompt)
    except Exception as e:
        response = f"ERROR: {e}"
    latency_ms = (time.time() - start_time) * 1000

    result = TestResult(
        model=model_name,
        source=test_case.get("source", "unknown"),
        task=test_case.get("task", "unknown"),
        difficulty=test_case.get("difficulty", "unknown"),
        raw_response=response,
        parsed_json=None,
        is_valid_json=False,
        latency_ms=latency_ms,
        used_kb=bool(kb_guidance)
    )

    # Verbose output
    if verbose:
        print(f"\n{'─'*60}")
        print(f"[PROMPT] {prompt[:100]}...")
        print(f"[RESPONSE] {response[:200]}...")

    # Try to parse JSON
    parsed, parse_error = extract_json_from_response(response)

    if parsed is None:
        result.errors.append(ValidationError(
            "parse_failure", "", "valid JSON", parse_error or "parse failed", "critical"
        ))
        return result

    result.parsed_json = parsed
    result.is_valid_json = True

    # Validate against schema
    result.errors.extend(validate_against_schema(parsed, schema))

    # Find hallucinated fields
    result.hallucinated_fields = find_hallucinated_fields(parsed, schema)
    for field_path in result.hallucinated_fields:
        result.errors.append(ValidationError(
            "hallucinated_field", field_path, "not in schema", "present", "major"
        ))

    if verbose and result.errors:
        print(f"[ERRORS] {[e.error_type for e in result.errors]}")

    return result


def run_test_suite(model_configs: List[dict], test_cases: List[dict],
                   iterations: int = 1, verbose: bool = False, use_kb: bool = False) -> List[TestResult]:
    """Run test suite with models loaded sequentially."""
    from model_runner import ModelRunner

    all_results = []
    total_models = len(model_configs)
    total_tests = len(test_cases) * iterations * total_models
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
                print(f"  KB guidance preview:\n{kb_guidance[:300]}...")
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

        # Create callable wrapper
        def make_model_call(r):
            def call(prompt: str) -> str:
                response, _ = r.generate(prompt, max_new_tokens=2048)
                return response
            return call

        model_func = make_model_call(runner)

        print(f"\n  Running {len(test_cases)} test cases x {iterations} iterations...")
        if kb_guidance:
            print(f"  📚 Using KB prompt guidance")

        for test_case in test_cases:
            source = test_case.get("source", "unknown")
            difficulty = test_case.get("difficulty", "?")

            for iteration in range(iterations):
                current_test += 1
                progress = f"[{current_test}/{total_tests}]"

                if not verbose:
                    print(f"  {progress} {source} ({difficulty})...", end=" ")

                result = run_single_test(
                    model_func, model_name, test_case,
                    verbose=verbose, kb_guidance=kb_guidance
                )
                all_results.append(result)

                if not verbose:
                    if not result.is_valid_json:
                        print(f"❌ PARSE FAIL ({result.latency_ms:.0f}ms)")
                    elif result.errors:
                        error_types = set(e.error_type for e in result.errors)
                        print(f"⚠️  {error_types} ({result.latency_ms:.0f}ms)")
                    else:
                        print(f"✅ OK ({result.latency_ms:.0f}ms)")

        runner.unload()

    return all_results


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(results: List[TestResult]) -> dict:
    """Generate analysis report matching paper metrics."""

    report = {
        "summary": {},
        "by_model": {},
        "by_difficulty": {},
        "by_source": {},
        "smoking_guns": []
    }

    if not results:
        return report

    # Check if KB was used
    used_kb = any(r.used_kb for r in results)

    # Per-model stats
    models = set(r.model for r in results)
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)

        parse_success = sum(1 for r in model_results if r.is_valid_json)
        full_compliance = sum(1 for r in model_results if r.is_valid_json and not r.errors)
        hallucinations = sum(1 for r in model_results if r.hallucinated_fields)

        error_breakdown = {}
        for r in model_results:
            for e in r.errors:
                error_breakdown[e.error_type] = error_breakdown.get(e.error_type, 0) + 1

        report["by_model"][model] = {
            "total_tests": total,
            "parse_success_rate": parse_success / total * 100 if total else 0,
            "full_compliance_rate": full_compliance / total * 100 if total else 0,
            "hallucination_rate": hallucinations / total * 100 if total else 0,
            "error_breakdown": error_breakdown,
            "avg_latency_ms": statistics.mean(r.latency_ms for r in model_results) if model_results else 0
        }

    # Per-difficulty stats (key for paper comparison)
    difficulties = set(r.difficulty for r in results)
    for diff in difficulties:
        diff_results = [r for r in results if r.difficulty == diff]
        total = len(diff_results)
        if total:
            report["by_difficulty"][diff] = {
                "total": total,
                "parse_success": sum(1 for r in diff_results if r.is_valid_json) / total * 100,
                "full_compliance": sum(1 for r in diff_results if r.is_valid_json and not r.errors) / total * 100,
                "hallucination_rate": sum(1 for r in diff_results if r.hallucinated_fields) / total * 100
            }

    # Per-source stats
    sources = set(r.source.split('/')[0] for r in results)
    for source in sources:
        source_results = [r for r in results if r.source.startswith(source)]
        total = len(source_results)
        if total:
            report["by_source"][source] = {
                "total": total,
                "parse_success": sum(1 for r in source_results if r.is_valid_json) / total * 100,
                "full_compliance": sum(1 for r in source_results if r.is_valid_json and not r.errors) / total * 100
            }

    # Find smoking guns
    for r in results:
        if r.is_valid_json and r.hallucinated_fields:
            report["smoking_guns"].append({
                "model": r.model,
                "source": r.source,
                "difficulty": r.difficulty,
                "description": "Valid JSON but invented fields",
                "hallucinated_fields": r.hallucinated_fields
            })

    # Overall summary
    total = len(results)
    report["summary"] = {
        "total_tests": total,
        "models_tested": list(models),
        "used_kb": used_kb,
        "overall_parse_success": sum(1 for r in results if r.is_valid_json) / total * 100 if total else 0,
        "overall_compliance": sum(1 for r in results if r.is_valid_json and not r.errors) / total * 100 if total else 0,
        "overall_hallucination": sum(1 for r in results if r.hallucinated_fields) / total * 100 if total else 0,
        "sources": list(sources),
        "total_smoking_guns": len(report["smoking_guns"])
    }

    return report


def print_report(report: dict):
    """Print formatted report."""

    print("\n" + "="*80)
    print("PAPER BENCHMARK RESULTS")
    print("="*80)

    s = report["summary"]
    kb_status = "WITH KB" if s.get("used_kb") else "BASELINE (no KB)"
    print(f"\n📊 SUMMARY ({kb_status})")
    print("-"*40)
    print(f"  Tests: {s['total_tests']} | Models: {len(s['models_tested'])}")
    print(f"  Parse Success: {s['overall_parse_success']:.1f}%")
    print(f"  Full Compliance: {s['overall_compliance']:.1f}%")
    print(f"  Hallucination Rate: {s['overall_hallucination']:.1f}%")
    print(f"  Smoking Gun Failures: {s['total_smoking_guns']}")

    print(f"\n📈 BY MODEL")
    print("-"*40)
    for model, stats in sorted(report["by_model"].items(),
                               key=lambda x: x[1]["full_compliance_rate"], reverse=True):
        print(f"\n  {model}")
        print(f"    Parse: {stats['parse_success_rate']:.1f}% | Comply: {stats['full_compliance_rate']:.1f}%")
        print(f"    Hallucinate: {stats['hallucination_rate']:.1f}% | Latency: {stats['avg_latency_ms']:.0f}ms")
        if stats["error_breakdown"]:
            print(f"    Errors: {stats['error_breakdown']}")

    print(f"\n📉 BY DIFFICULTY (Paper Comparison)")
    print("-"*40)
    difficulty_order = ["easy", "medium", "hard", "ultra"]
    for diff in difficulty_order:
        if diff in report["by_difficulty"]:
            stats = report["by_difficulty"][diff]
            print(f"  {diff:8} Parse: {stats['parse_success']:.0f}% Comply: {stats['full_compliance']:.0f}% Halluc: {stats['hallucination_rate']:.0f}%")

    print(f"\n📚 BY SOURCE")
    print("-"*40)
    for source, stats in report["by_source"].items():
        print(f"  {source:20} Parse: {stats['parse_success']:.0f}% Comply: {stats['full_compliance']:.0f}%")

    if report["smoking_guns"]:
        print(f"\n🚨 SMOKING GUN FAILURES (first 5)")
        print("-"*40)
        for i, sg in enumerate(report["smoking_guns"][:5], 1):
            print(f"  {i}. [{sg['model']}] {sg['source']} ({sg['difficulty']})")
            print(f"     Hallucinated: {sg['hallucinated_fields']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper Benchmark Test Suite")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("-i", "--iterations", type=int, default=1,
                        help="Number of iterations per test case (default: 1)")
    parser.add_argument("--models", type=str, default="models.json",
                        help="Path to models config file (default: models.json)")
    parser.add_argument("--kb", action="store_true",
                        help="Enable knowledge base prompt guidance")
    parser.add_argument("--no-kb", action="store_true",
                        help="Disable knowledge base (default)")
    args = parser.parse_args()

    print("="*80)
    print("PAPER BENCHMARK TEST SUITE")
    print("Using test cases from: JSONSchemaBench, SchemaBench, StructuredRAG")
    print("="*80)

    # Load model configs
    models_path = args.models
    if not os.path.exists(models_path):
        models_path = os.path.join(os.path.dirname(__file__), args.models)

    if not os.path.exists(models_path):
        print(f"\n❌ Models config not found: {args.models}")
        print("Create a models.json file with your model configurations.")
        return None

    with open(models_path, 'r') as f:
        model_configs = json.load(f)

    if isinstance(model_configs, dict) and "models" in model_configs:
        model_configs = model_configs["models"]

    model_names = [c.get("display_name", c.get("name")) for c in model_configs]
    kb_status = "ENABLED" if args.kb else "DISABLED"

    print(f"\nModels: {model_names}")
    print(f"Test cases: {len(ALL_TEST_CASES)}")
    print(f"Iterations: {args.iterations}")
    print(f"Knowledge Base: {kb_status}")

    print(f"\nTest breakdown:")
    print(f"  SchemaBench: {len(SCHEMABENCH_TESTS)} tests")
    print(f"  StructuredRAG: {len(STRUCTUREDRAG_TESTS)} tests")
    print(f"  JSONSchemaBench: {len(JSONSCHEMABENCH_TESTS)} tests")

    # Run tests
    print("\nStarting tests...")
    results = run_test_suite(
        model_configs,
        ALL_TEST_CASES,
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

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kb_suffix = "_kb" if args.kb else "_baseline"
    filename = f"paper_benchmark{kb_suffix}_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), "results", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    output = {
        "metadata": {
            "timestamp": timestamp,
            "used_kb": args.kb,
            "iterations": args.iterations,
            "models": model_names
        },
        "report": report,
        "raw_results": [
            {
                "model": r.model,
                "source": r.source,
                "task": r.task,
                "difficulty": r.difficulty,
                "is_valid_json": r.is_valid_json,
                "errors": [
                    {"type": e.error_type, "path": e.field_path,
                     "expected": e.expected, "actual": e.actual}
                    for e in r.errors
                ],
                "hallucinated_fields": r.hallucinated_fields,
                "latency_ms": r.latency_ms,
                "used_kb": r.used_kb,
                "raw_response": r.raw_response[:500]
            }
            for r in results
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Also save as latest
    latest_path = os.path.join(os.path.dirname(filepath), f"paper_benchmark{kb_suffix}_{timestamp}_latest.json")
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {filepath}")
    print(f"Latest results at: {latest_path}")

    return report


if __name__ == "__main__":
    main()
