"""Privacy types for Valence belief sharing.

Implements SharePolicy with graduated sharing levels and enforcement types.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ShareLevel(Enum):
    """Graduated levels of belief sharing permissions."""

    PRIVATE = "private"  # Never leaves node
    DIRECT = "direct"  # Specific recipient, no reshare
    BOUNDED = "bounded"  # Can reshare within scope
    CASCADING = "cascading"  # Propagates with restrictions
    PUBLIC = "public"  # Open


class EnforcementType(Enum):
    """How sharing policies are enforced."""

    CRYPTOGRAPHIC = "cryptographic"  # Math enforced (encryption, signatures)
    POLICY = "policy"  # Protocol enforced (software checks)
    HONOR = "honor"  # Trust-based (no technical enforcement)


@dataclass
class PropagationRules:
    """Rules governing how beliefs can propagate through the network."""

    max_hops: int | None = None
    allowed_domains: list[str] | None = None
    min_trust_to_receive: float | None = None
    strip_on_forward: list[str] | None = None  # Fields to remove on forward
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_hops": self.max_hops,
            "allowed_domains": self.allowed_domains,
            "min_trust_to_receive": self.min_trust_to_receive,
            "strip_on_forward": self.strip_on_forward,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PropagationRules":
        """Deserialize from dictionary."""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            max_hops=data.get("max_hops"),
            allowed_domains=data.get("allowed_domains"),
            min_trust_to_receive=data.get("min_trust_to_receive"),
            strip_on_forward=data.get("strip_on_forward"),
            expires_at=expires_at,
        )


@dataclass
class SharePolicy:
    """Policy controlling how a belief can be shared.

    Combines a ShareLevel with enforcement type and optional propagation rules.
    """

    level: ShareLevel
    enforcement: EnforcementType = EnforcementType.POLICY
    recipients: list[str] | None = None  # DIDs for DIRECT sharing
    propagation: PropagationRules | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "level": self.level.value,
            "enforcement": self.enforcement.value,
            "recipients": self.recipients,
            "propagation": self.propagation.to_dict() if self.propagation else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharePolicy":
        """Deserialize from dictionary."""
        return cls(
            level=ShareLevel(data["level"]),
            enforcement=EnforcementType(data.get("enforcement", "policy")),
            recipients=data.get("recipients"),
            propagation=(PropagationRules.from_dict(data["propagation"]) if data.get("propagation") else None),
        )

    @classmethod
    def private(cls) -> "SharePolicy":
        """Create a private policy - belief never leaves the node."""
        return cls(level=ShareLevel.PRIVATE, enforcement=EnforcementType.CRYPTOGRAPHIC)

    @classmethod
    def public(cls) -> "SharePolicy":
        """Create a public policy - belief is openly shareable."""
        return cls(level=ShareLevel.PUBLIC, enforcement=EnforcementType.HONOR)

    @classmethod
    def direct(cls, recipients: list[str]) -> "SharePolicy":
        """Create a direct policy - share only with specific recipients."""
        return cls(
            level=ShareLevel.DIRECT,
            enforcement=EnforcementType.CRYPTOGRAPHIC,
            recipients=recipients,
        )

    @classmethod
    def bounded(cls, max_hops: int = 2, allowed_domains: list[str] | None = None) -> "SharePolicy":
        """Create a bounded policy - can reshare within scope."""
        return cls(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=max_hops,
                allowed_domains=allowed_domains,
            ),
        )

    def allows_sharing_to(self, recipient_did: str) -> bool:
        """Check if this policy allows sharing to a specific recipient."""
        if self.level == ShareLevel.PRIVATE:
            return False
        if self.level == ShareLevel.PUBLIC:
            return True
        if self.level == ShareLevel.DIRECT:
            return self.recipients is not None and recipient_did in self.recipients
        # BOUNDED and CASCADING require additional context (hop count, trust, etc.)
        return True

    def is_expired(self) -> bool:
        """Check if the policy has expired."""
        if self.propagation and self.propagation.expires_at:
            now = datetime.now(UTC).replace(tzinfo=None)
            return now > self.propagation.expires_at
        return False


class SharingIntent(Enum):
    """High-level sharing intents that map to SharePolicy configurations.

    These represent the *why* of sharing, not just the mechanics:
    - know_me: Share identity/context with a specific trusted person
    - work_with_me: Collaborate with a bounded group
    - learn_from_me: Publish knowledge for others to build on
    - use_this: Make something freely available
    """

    KNOW_ME = "know_me"
    WORK_WITH_ME = "work_with_me"
    LEARN_FROM_ME = "learn_from_me"
    USE_THIS = "use_this"


# Default max_hops per intent (None = unlimited)
_INTENT_DEFAULT_MAX_HOPS: dict[SharingIntent, int | None] = {
    SharingIntent.KNOW_ME: 0,
    SharingIntent.WORK_WITH_ME: 2,
    SharingIntent.LEARN_FROM_ME: None,
    SharingIntent.USE_THIS: None,
}


@dataclass
class IntentConfig:
    """Configuration that pairs a SharingIntent with the generated SharePolicy.

    Preserves the user's original sharing decision (the intent) alongside
    the mechanical policy it produces. This lets us show "you shared this
    as know_me" rather than just "DIRECT + CRYPTOGRAPHIC + max_hops=0".

    Args:
        intent: The high-level sharing intent
        recipients: Required for know_me, optional for others
        max_hops: Override default max_hops for the intent
        expires_at: Optional expiration for the share
    """

    intent: SharingIntent
    recipients: list[str] | None = None
    max_hops: int | None = None  # None means use intent default
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.intent == SharingIntent.KNOW_ME and not self.recipients:
            raise ValueError("know_me intent requires at least one recipient")

    def to_share_policy(self) -> SharePolicy:
        """Generate the appropriate SharePolicy for this intent."""
        effective_max_hops = self.max_hops if self.max_hops is not None else _INTENT_DEFAULT_MAX_HOPS[self.intent]
        propagation = (
            PropagationRules(
                max_hops=effective_max_hops,
                expires_at=self.expires_at,
            )
            if effective_max_hops is not None or self.expires_at is not None
            else None
        )

        if self.intent == SharingIntent.KNOW_ME:
            return SharePolicy(
                level=ShareLevel.DIRECT,
                enforcement=EnforcementType.CRYPTOGRAPHIC,
                recipients=self.recipients,
                propagation=propagation,
            )
        elif self.intent == SharingIntent.WORK_WITH_ME:
            return SharePolicy(
                level=ShareLevel.BOUNDED,
                enforcement=EnforcementType.POLICY,
                recipients=self.recipients,
                propagation=propagation,
            )
        elif self.intent == SharingIntent.LEARN_FROM_ME:
            return SharePolicy(
                level=ShareLevel.CASCADING,
                enforcement=EnforcementType.POLICY,
                propagation=propagation,
            )
        else:  # USE_THIS
            return SharePolicy(
                level=ShareLevel.PUBLIC,
                enforcement=EnforcementType.HONOR,
                propagation=propagation,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSONB storage."""
        return {
            "intent": self.intent.value,
            "recipients": self.recipients,
            "max_hops": self.max_hops,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "policy": self.to_share_policy().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentConfig":
        """Deserialize from dictionary."""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            intent=SharingIntent(data["intent"]),
            recipients=data.get("recipients"),
            max_hops=data.get("max_hops"),
            expires_at=expires_at,
        )
