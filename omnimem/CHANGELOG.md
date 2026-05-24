# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-30

### Added
- Initial release of OmniMem
- Five-layer memory architecture (L0 Perception → L1 Working → L2 Structured → L3 Deep → L4 Internalized)
- Hybrid retrieval engine (Vector + BM25 + RRF fusion + Knowledge Graph)
- Complete governance engine (Conflict resolution, Temporal decay, Forgetting curve, Privacy levels, Provenance tracking)
- Saga transaction coordination for derived data consistency
- Multi-instance synchronization with vector clocks
- Built-in memory tool compatibility layer
- Context Manager with semantic deduplication and token budget control
- Security features (anti-recursion, input sanitization, Unicode normalization)
- Comprehensive test suite covering all core modules
- `pyproject.toml` for modern Python packaging and dependency management
- `requirements.txt` and `requirements-dev.txt` for dependency installation
- GitHub Actions CI workflow for automated testing and linting
- Restructured tests into `tests/` directory with proper pytest configuration
- Pre-commit hooks configuration (ruff, mypy, trailing-whitespace)
- GitHub Issue templates (bug report, feature request)
- GitHub Pull Request template
- Social preview generator

### Changed
- Tests now use `omnimem.*` imports instead of `plugins.memory.omnimem.*`

### Fixed
- Unified import paths from `plugins.memory.omnimem` to `omnimem`
- Upgraded GitHub Actions to Node.js 24 (checkout v5, setup-python v6, codecov-action v5)
