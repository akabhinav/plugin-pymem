"""Tests for scope enforcement — cross-scope access must fail."""

import pytest

from pymem.models import MemoryScope


class TestScopeEnforcement:
    def test_empty_scope_fails(self):
        with pytest.raises(ValueError):
            MemoryScope()

    def test_user_scope_valid(self):
        scope = MemoryScope(user_id="u1")
        assert scope.user_id == "u1"

    def test_agent_scope_valid(self):
        scope = MemoryScope(agent_id="a1")
        assert scope.agent_id == "a1"

    def test_session_scope_valid(self):
        scope = MemoryScope(session_id="s1")
        assert scope.session_id == "s1"

    def test_org_scope_valid(self):
        scope = MemoryScope(org_id="o1")
        assert scope.org_id == "o1"

    def test_scope_filter_excludes_none(self):
        scope = MemoryScope(user_id="u1")
        f = scope.to_filter()
        assert "agent_id" not in f
        assert "session_id" not in f
