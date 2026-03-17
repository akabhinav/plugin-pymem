"""Tests for the decay engine — forgetting curve calculations."""

import pytest

from pymem.engine.decay_engine import DecayEngine
from pymem.models import MemoryType


class TestDecayEngine:
    def setup_method(self):
        self.decay = DecayEngine()

    def test_semantic_decay(self):
        # After 100 days, score should decay
        score = self.decay.calculate_score(1.0, 100, MemoryType.SEMANTIC)
        assert 0.3 < score < 0.5  # exp(-0.01 * 100) ≈ 0.368

    def test_episodic_decay_slower(self):
        # Episodic decays slower than semantic
        sem_score = self.decay.calculate_score(1.0, 100, MemoryType.SEMANTIC)
        epi_score = self.decay.calculate_score(1.0, 100, MemoryType.EPISODIC)
        assert epi_score > sem_score

    def test_procedural_never_decays(self):
        score = self.decay.calculate_score(0.8, 1000, MemoryType.PROCEDURAL)
        assert score == 0.8

    def test_working_never_decays(self):
        score = self.decay.calculate_score(0.5, 500, MemoryType.WORKING)
        assert score == 0.5

    def test_zero_days_no_decay(self):
        score = self.decay.calculate_score(0.9, 0, MemoryType.SEMANTIC)
        assert score == pytest.approx(0.9)

    def test_should_soft_delete(self):
        assert self.decay.should_soft_delete(0.05) is True
        assert self.decay.should_soft_delete(0.15) is False

    def test_should_downrank(self):
        assert self.decay.should_downrank(0.2) is True
        assert self.decay.should_downrank(0.5) is False
