# Contributing to stakeout-agent

Thank you for contributing to `stakeout-agent`.

`stakeout-agent` is a lightweight monitoring package for LangGraph applications. Contributions should keep the package simple, reliable, and easy to integrate.

## Principles

Please follow these principles:

- Keep the library easy to use and install.
- Monitoring should not break the user's application.
- Keep framework-specific code separated from core monitoring logic.
- Avoid storing sensitive or excessive data.
- Prefer simple, readable code over clever abstractions.

## Development Setup

Clone the repository:

```bash
git clone https://github.com/KyriakosFrang/stakeout-agent.git
cd stakeout-agent
```

Install dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

Run linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

## Code Standards

- Use Python 3.10+.
- Keep functions small and focused.
- Use clear names.
- Add type hints where useful.
- Handle errors defensively.
- Do not let monitoring/database failures crash the user application.
- Avoid introducing large dependencies unless clearly justified.

## Testing Standards

Add or update tests when changing behavior.

Tests are expected for:

- callback behavior
- database writes
- error handling
- serialization logic
- new public APIs
- bug fixes

Before opening a pull request, run:

```bash
uv run pytest
uv run ruff check .
```

## Security and Privacy

This package may capture prompts, messages, tool calls, and agent state.

Do not store or log:

- API keys
- passwords
- tokens
- connection strings
- private customer data
- large unbounded payloads

When capturing messages or payloads, keep them bounded, truncated, and documented.

## Pull Request Checklist

Before submitting a PR, make sure:

- [ ] The change has a clear purpose
- [ ] Tests were added or updated if needed
- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` passes
- [ ] Documentation was updated if behavior changed
- [ ] No secrets or sensitive data were committed

## Commit Messages

Use clear commit messages.

Examples:

```text
feat: capture LangGraph messages
fix: handle MongoDB write failures
docs: update installation instructions
test: add callback handler tests
refactor: simplify event serialization
```

## Documentation

Update the README when changing:

- installation
- public imports
- callback usage
- configuration
- MongoDB behavior
- examples

## Definition of Done

A contribution is done when:

- the code works
- the behavior is tested
- formatting and linting pass
- documentation is updated where needed
- the package remains simple and safe to use
