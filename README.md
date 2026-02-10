# our-privacy

Trust graphs, capability-based access control, audit trails, watermarking, and GDPR-compliant data export for the ourochronos ecosystem.

## Overview

our-privacy is a comprehensive privacy and trust management system. It implements multi-dimensional trust relationships, OCAP-style authorization with short-lived bearer tokens, tamper-evident audit logging, graduated sharing levels (private to public), invisible watermarking for leak detection, and self-service GDPR data export. It provides the privacy guarantees that make federated knowledge sharing safe.

## Install

```bash
pip install our-privacy
```

Requires `our-db>=0.1.0`, `cryptography>=41.0`, and `PyJWT>=2.8`.

## Usage

### Trust Management

```python
from our_privacy import TrustEdge4D, TrustService

# 4-dimensional trust: competence, integrity, confidentiality, judgment
edge = TrustEdge4D(
    source_did="did:key:alice",
    target_did="did:key:bob",
    competence=0.9,
    integrity=0.8,
    confidentiality=0.7,
    judgment=0.6,
)
```

### Capabilities

```python
from our_privacy import issue_capability, verify_capability, CapabilityAction

# Issue a short-lived bearer token
cap = issue_capability(
    issuer_did="did:valence:issuer",
    holder_did="did:valence:user",
    resource="valence://beliefs/tech",
    actions=[CapabilityAction.READ, CapabilityAction.SHARE],
    ttl_seconds=900,  # 15 minutes
)

# Verify before granting access
verify_capability(cap, holder_did, resource, CapabilityAction.READ)
```

### Sharing Policies

```python
from our_privacy import SharePolicy

# Graduated sharing levels
policy = SharePolicy.private()                           # Only owner
policy = SharePolicy.direct(recipients=["did:key:bob"])  # Specific recipients
policy = SharePolicy.bounded(max_hops=2)                 # Limited propagation
policy = SharePolicy.public()                            # Open access
```

### Audit Logging

```python
from our_privacy import get_audit_logger, verify_chain

logger = get_audit_logger()
logger.log_event(event_type, actor, action, resource)

# Tamper-evident: SHA-256 hash chain
events = logger.get_events()
verify_chain(events)  # Raises ChainVerificationError if tampered
```

### Watermarking

```python
from our_privacy import embed_watermark, extract_watermark, WatermarkTechnique

# Invisible watermark for leak tracing
watermarked = embed_watermark(
    content="Sensitive report content",
    recipient_id="user123",
    technique=WatermarkTechnique.WHITESPACE,
)

# If content leaks, identify the source
watermark = extract_watermark(watermarked)
```

### Data Export (GDPR)

```python
from our_privacy import generate_data_report, ReportScope

scope = ReportScope(
    include_beliefs=True,
    include_shares_sent=True,
    include_audit_events=True,
)
report = await generate_data_report(user_did, scope, format="json")
```

## API

### Trust

`TrustEdge`, `TrustEdge4D`, `TrustService`, `TrustGraphStore`, `DecayModel`

### Capabilities

`Capability`, `CapabilityAction` (READ, WRITE, DELETE, SHARE, DELEGATE, ADMIN, QUERY, EMBED, FEDERATE), `CapabilityService`, `issue_capability()`, `verify_capability()`, `revoke_capability()`, `requires_capability` (decorator)

### Sharing

`SharePolicy`, `ShareLevel` (PRIVATE, DIRECT, BOUNDED, CASCADING, PUBLIC), `SharingService`, `PropagationRules`

### Audit

`AuditLogger`, `AuditEvent`, `AuditEventType`, `AuditBackend`, `InMemoryAuditBackend`, `FileAuditBackend`, `verify_chain()`

### Watermarking & Canaries

`embed_watermark()`, `extract_watermark()`, `WatermarkTechnique` (WHITESPACE, HOMOGLYPH, COMBINED), `CanaryToken`, `embed_canary()`, `detect_canaries()`

### Domains & Elevation

`Domain`, `DomainService`, `DomainRole`, `ElevationProposal`, `ElevationService`

### Reports

`generate_data_report()`, `DataReport`, `ReportScope`, `ExportFormat` (JSON, CSV)

### Additional

`CorroborationDetector`, `AnomalyDetector`, `EncryptionEnvelope`, `ProvenanceChain`

## Key Properties

- **4D trust model**: Competence, integrity, confidentiality, judgment — with time-based decay
- **Hash-chain audit**: Tamper-evident SHA-256 chain with PII sanitization
- **OCAP authorization**: Unforgeable, short-lived (15min default), revocable tokens
- **Graduated sharing**: Five levels from private to public with policy or cryptographic enforcement
- **Leak tracing**: Invisible watermarks and canary tokens for detecting unauthorized sharing
- **1,135 tests** covering trust, capabilities, audit, sharing, watermarking, and GDPR compliance

## Development

```bash
# Install with dev dependencies
make dev

# Run linters
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Auto-format
make format
```

## State Ownership

Owns trust edges, capabilities, audit events, sharing policies, watermark records, and domain memberships. Storage is pluggable via backend interfaces.

## Part of Valence

This brick is part of the [Valence](https://github.com/ourochronos/valence) knowledge substrate. See [our-infra](https://github.com/ourochronos/our-infra) for ourochronos conventions.

## License

MIT
