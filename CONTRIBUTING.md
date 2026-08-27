# Contributing to llm-saia

## Development Setup

```bash
git clone https://github.com/llm-works/llm-saia.git
cd llm-saia

python -m pip install -e ".[dev]"
```

## Running Tests

```bash
make test           # unit tests
make check          # format, lint, type check, tests
```

## Sign-off (DCO)

We use the [Developer Certificate of Origin (DCO)](https://developercertificate.org). Every commit
must be signed off, asserting you have the right to submit it under the project's Apache-2.0
license.

Add sign-off automatically:

```bash
git commit -s -m "your commit message"
```

This appends `Signed-off-by: Your Name <your@email>` to the commit message. Forgot? Amend:

```bash
git commit --amend -s
```

## PR Process

- Fork the repo, work on a branch
- Ensure every commit has `Signed-off-by:` (automated check will fail if missing)
- Open PR against `develop`
- Maintainer review + merge gate applies — not every PR will be merged

**Merge policy:** all PRs are squash-merged.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
