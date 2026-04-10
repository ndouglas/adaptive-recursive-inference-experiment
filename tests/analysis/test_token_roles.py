"""Tests for TokenRoleClassifier."""
import pytest

from src.analysis.token_roles import TokenRoleClassifier, TokenRole


class TestTokenRoleClassifier:
    @pytest.fixture
    def classifier(self):
        return TokenRoleClassifier("Qwen/Qwen2.5-1.5B")

    def test_classify_simple_json(self, classifier):
        text = '{"reasoning": "2 + 3 = 5", "answer": 5}'
        roles = classifier.classify(text)
        assert len(roles) > 0
        assert roles[0] == TokenRole.STRUCTURAL
        assert roles[-1] == TokenRole.STRUCTURAL
        role_set = set(roles)
        assert TokenRole.STRUCTURAL in role_set
        assert TokenRole.REASONING in role_set
        assert TokenRole.ANSWER in role_set

    def test_structural_tokens_identified(self, classifier):
        text = '{"reasoning": "yes", "answer": 42}'
        roles = classifier.classify(text)
        structural_count = sum(1 for r in roles if r == TokenRole.STRUCTURAL)
        assert structural_count >= 4

    def test_answer_tokens_at_end(self, classifier):
        text = '{"reasoning": "the answer is 7", "answer": 7}'
        roles = classifier.classify(text)
        last_reasoning_idx = max(
            i for i, r in enumerate(roles) if r == TokenRole.REASONING
        )
        first_answer_idx = min(
            i for i, r in enumerate(roles) if r == TokenRole.ANSWER
        )
        assert first_answer_idx > last_reasoning_idx

    def test_returns_correct_length(self, classifier):
        text = '{"reasoning": "test content here", "answer": 123}'
        roles = classifier.classify(text)
        tokenizer = classifier.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(roles) == len(token_ids)

    def test_role_enum_values(self):
        assert TokenRole.STRUCTURAL.value == "structural"
        assert TokenRole.REASONING.value == "reasoning"
        assert TokenRole.ANSWER.value == "answer"
