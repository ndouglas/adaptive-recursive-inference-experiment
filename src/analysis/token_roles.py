"""Classify generated tokens by their role in constrained JSON output.

The generated text has the structure:
    {"reasoning": "<work>", "answer": <number>}

Each token is classified as:
- STRUCTURAL: JSON syntax (braces, keys, colons, commas, quotes around fields)
- REASONING: content inside the "reasoning" value string
- ANSWER: the numeric answer value tokens
"""
from enum import Enum

from transformers import AutoTokenizer


class TokenRole(Enum):
    STRUCTURAL = "structural"
    REASONING = "reasoning"
    ANSWER = "answer"


class TokenRoleClassifier:
    """Classify tokens in constrained JSON output by their semantic role.

    Uses the Qwen tokenizer to re-tokenize generated text and map each
    token to a JSON role based on character position.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def classify(self, generated_text):
        """Classify each token in generated_text by JSON role.

        Args:
            generated_text: The full generated JSON string,
                e.g. '{"reasoning": "work here", "answer": 42}'

        Returns:
            List of TokenRole, one per token.
        """
        token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)

        reasoning_start, reasoning_end = self._find_reasoning_span(generated_text)
        answer_start, answer_end = self._find_answer_span(generated_text)

        char_spans = self._token_char_spans(token_ids)

        roles = []
        for start, end in char_spans:
            mid = (start + end) // 2
            if reasoning_start <= mid < reasoning_end:
                roles.append(TokenRole.REASONING)
            elif answer_start <= mid < answer_end:
                roles.append(TokenRole.ANSWER)
            else:
                roles.append(TokenRole.STRUCTURAL)

        return roles

    def _find_reasoning_span(self, text):
        """Find char range of the reasoning value content (inside quotes)."""
        key = '"reasoning":'
        idx = text.find(key)
        if idx == -1:
            return (0, 0)
        after_key = idx + len(key)
        quote_start = text.find('"', after_key)
        if quote_start == -1:
            return (0, 0)
        pos = quote_start + 1
        while pos < len(text):
            if text[pos] == '\\':
                pos += 2
                continue
            if text[pos] == '"':
                break
            pos += 1
        return (quote_start + 1, pos)

    def _find_answer_span(self, text):
        """Find char range of the answer value (the number after "answer":)."""
        key = '"answer":'
        idx = text.rfind(key)
        if idx == -1:
            return (0, 0)
        after_key = idx + len(key)
        pos = after_key
        while pos < len(text) and text[pos] == ' ':
            pos += 1
        end = pos
        while end < len(text) and text[end] not in ('}', ','):
            end += 1
        return (pos, end)

    def _token_char_spans(self, token_ids):
        """Compute (start, end) character span for each token."""
        spans = []
        prev_len = 0
        for i in range(len(token_ids)):
            decoded = self.tokenizer.decode(token_ids[:i + 1])
            cur_len = len(decoded)
            spans.append((prev_len, cur_len))
            prev_len = cur_len
        return spans
