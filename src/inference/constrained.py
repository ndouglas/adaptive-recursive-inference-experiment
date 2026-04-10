"""Constrained decoding via outlines for structured JSON output.

Builds a logits processor that forces the model to produce valid JSON
matching a given schema. Used with AdaptiveLoop.generate(logits_processor=...).
"""
import json

import outlines
from outlines.backends import get_json_schema_logits_processor


MATH_ANSWER_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "number"},
    },
    "required": ["reasoning", "answer"],
})


def build_json_processor(model, tokenizer, schema=MATH_ANSWER_SCHEMA):
    """Build an outlines logits processor for JSON-constrained decoding.

    Args:
        model: HuggingFace AutoModelForCausalLM
        tokenizer: HuggingFace tokenizer
        schema: JSON schema string

    Returns:
        logits_processor: callable(input_ids, logits) -> logits
            Call .reset() before each new sequence.
    """
    wrapped = outlines.from_transformers(model, tokenizer)
    return get_json_schema_logits_processor(
        backend_name=None,
        model=wrapped,
        json_schema=schema,
    )
