"""Tests for SharingIntent and IntentConfig."""

from datetime import datetime

import pytest

from our_privacy.types import (
    EnforcementType,
    IntentConfig,
    ShareLevel,
    SharingIntent,
)


class TestSharingIntent:
    """Tests for SharingIntent enum."""

    def test_all_intents_exist(self):
        """Verify all 4 sharing intents are defined."""
        assert SharingIntent.KNOW_ME.value == "know_me"
        assert SharingIntent.WORK_WITH_ME.value == "work_with_me"
        assert SharingIntent.LEARN_FROM_ME.value == "learn_from_me"
        assert SharingIntent.USE_THIS.value == "use_this"

    def test_intent_from_string(self):
        """Test creating intent from string value."""
        assert SharingIntent("know_me") == SharingIntent.KNOW_ME
        assert SharingIntent("use_this") == SharingIntent.USE_THIS

    def test_exactly_four_intents(self):
        """Ensure no extra intents have been added without tests."""
        assert len(SharingIntent) == 4


class TestIntentConfigKnowMe:
    """Tests for know_me intent."""

    def test_requires_recipients(self):
        """know_me without recipients raises ValueError."""
        with pytest.raises(ValueError, match="know_me intent requires at least one recipient"):
            IntentConfig(intent=SharingIntent.KNOW_ME)

    def test_requires_recipients_empty_list(self):
        """know_me with empty recipients list raises ValueError."""
        with pytest.raises(ValueError, match="know_me intent requires at least one recipient"):
            IntentConfig(intent=SharingIntent.KNOW_ME, recipients=[])

    def test_to_share_policy(self):
        """know_me produces DIRECT + CRYPTOGRAPHIC with max_hops=0."""
        config = IntentConfig(intent=SharingIntent.KNOW_ME, recipients=["did:key:alice"])
        policy = config.to_share_policy()

        assert policy.level == ShareLevel.DIRECT
        assert policy.enforcement == EnforcementType.CRYPTOGRAPHIC
        assert policy.recipients == ["did:key:alice"]
        assert policy.propagation is not None
        assert policy.propagation.max_hops == 0

    def test_multiple_recipients(self):
        """know_me supports multiple recipients."""
        recipients = ["did:key:alice", "did:key:bob"]
        config = IntentConfig(intent=SharingIntent.KNOW_ME, recipients=recipients)
        policy = config.to_share_policy()

        assert policy.recipients == recipients


class TestIntentConfigWorkWithMe:
    """Tests for work_with_me intent."""

    def test_to_share_policy(self):
        """work_with_me produces BOUNDED + POLICY with max_hops=2."""
        config = IntentConfig(intent=SharingIntent.WORK_WITH_ME, recipients=["did:key:alice"])
        policy = config.to_share_policy()

        assert policy.level == ShareLevel.BOUNDED
        assert policy.enforcement == EnforcementType.POLICY
        assert policy.recipients == ["did:key:alice"]
        assert policy.propagation is not None
        assert policy.propagation.max_hops == 2

    def test_no_recipients_required(self):
        """work_with_me does not require recipients."""
        config = IntentConfig(intent=SharingIntent.WORK_WITH_ME)
        policy = config.to_share_policy()
        assert policy.level == ShareLevel.BOUNDED

    def test_custom_max_hops(self):
        """work_with_me allows overriding max_hops."""
        config = IntentConfig(intent=SharingIntent.WORK_WITH_ME, max_hops=5)
        policy = config.to_share_policy()
        assert policy.propagation.max_hops == 5


class TestIntentConfigLearnFromMe:
    """Tests for learn_from_me intent."""

    def test_to_share_policy(self):
        """learn_from_me produces CASCADING + POLICY with no hop limit."""
        config = IntentConfig(intent=SharingIntent.LEARN_FROM_ME)
        policy = config.to_share_policy()

        assert policy.level == ShareLevel.CASCADING
        assert policy.enforcement == EnforcementType.POLICY
        assert policy.propagation is None  # No propagation rules when unlimited + no expiry

    def test_with_expiry(self):
        """learn_from_me with expiry generates propagation rules."""
        expires = datetime(2026, 12, 31)
        config = IntentConfig(intent=SharingIntent.LEARN_FROM_ME, expires_at=expires)
        policy = config.to_share_policy()

        assert policy.propagation is not None
        assert policy.propagation.max_hops is None
        assert policy.propagation.expires_at == expires


class TestIntentConfigUseThis:
    """Tests for use_this intent."""

    def test_to_share_policy(self):
        """use_this produces PUBLIC + HONOR with no restrictions."""
        config = IntentConfig(intent=SharingIntent.USE_THIS)
        policy = config.to_share_policy()

        assert policy.level == ShareLevel.PUBLIC
        assert policy.enforcement == EnforcementType.HONOR
        assert policy.propagation is None


class TestIntentConfigSerialization:
    """Tests for IntentConfig to_dict/from_dict."""

    def test_to_dict_know_me(self):
        """to_dict includes intent, recipients, and generated policy."""
        config = IntentConfig(intent=SharingIntent.KNOW_ME, recipients=["did:key:alice"])
        data = config.to_dict()

        assert data["intent"] == "know_me"
        assert data["recipients"] == ["did:key:alice"]
        assert data["max_hops"] is None  # Uses default, not stored
        assert data["expires_at"] is None
        assert data["policy"]["level"] == "direct"
        assert data["policy"]["enforcement"] == "cryptographic"

    def test_to_dict_with_expiry(self):
        """to_dict serializes expiry as ISO format."""
        expires = datetime(2026, 6, 15, 12, 0, 0)
        config = IntentConfig(intent=SharingIntent.USE_THIS, expires_at=expires)
        data = config.to_dict()

        assert data["expires_at"] == "2026-06-15T12:00:00"

    def test_from_dict_roundtrip_know_me(self):
        """from_dict round-trips know_me config."""
        original = IntentConfig(intent=SharingIntent.KNOW_ME, recipients=["did:key:alice", "did:key:bob"])
        restored = IntentConfig.from_dict(original.to_dict())

        assert restored.intent == original.intent
        assert restored.recipients == original.recipients
        assert restored.max_hops == original.max_hops
        assert restored.expires_at == original.expires_at

    def test_from_dict_roundtrip_work_with_me(self):
        """from_dict round-trips work_with_me config with custom max_hops."""
        original = IntentConfig(intent=SharingIntent.WORK_WITH_ME, max_hops=3)
        restored = IntentConfig.from_dict(original.to_dict())

        assert restored.intent == SharingIntent.WORK_WITH_ME
        assert restored.max_hops == 3

    def test_from_dict_roundtrip_with_expiry(self):
        """from_dict round-trips expiry timestamp."""
        expires = datetime(2026, 12, 31, 23, 59, 59)
        original = IntentConfig(intent=SharingIntent.LEARN_FROM_ME, expires_at=expires)
        restored = IntentConfig.from_dict(original.to_dict())

        assert restored.expires_at == expires

    def test_from_dict_minimal(self):
        """from_dict works with minimal data."""
        data = {"intent": "use_this"}
        config = IntentConfig.from_dict(data)
        assert config.intent == SharingIntent.USE_THIS
        assert config.recipients is None
        assert config.max_hops is None

    def test_policy_preserved_in_dict(self):
        """The generated policy is included in to_dict for JSONB queries."""
        config = IntentConfig(intent=SharingIntent.WORK_WITH_ME, recipients=["did:key:alice"])
        data = config.to_dict()

        assert "policy" in data
        assert data["policy"]["level"] == "bounded"
        assert data["policy"]["enforcement"] == "policy"
