# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

infra := $(shell appinfra scripts-path)

# Configuration
APPINFRA_DEV_PKG_NAME := llm_saia

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
APPINFRA_DEV_CQ_STRICT := true

# Include SPDX header check in `make check`.
APPINFRA_DEV_CQ_SPDX := true

# No ruff rules reported as warnings only: they are enforced in pyproject's select
APPINFRA_DEV_CQ_LINT_SELECT := false

# Fail on duplicated code blocks and unused private definitions
APPINFRA_DEV_CQ_DUP := strict
APPINFRA_DEV_CQ_DEAD := strict

# Report tests that sit in another suite's directory, or leak process-wide state
APPINFRA_DEV_TEST_HEALTH := warn

# Run every example script as part of `make check`
APPINFRA_DEV_CHECK_EXAMPLES := true

# Test coverage threshold (percentage), measured over unit and integration tests
APPINFRA_PYTEST_COVERAGE_THRESHOLD := 95
APPINFRA_PYTEST_COVERAGE_MARKERS := unit or integration

# Docstring coverage threshold (percentage)
APPINFRA_DEV_DOCSTRING_THRESHOLD := 90

# Include framework (config first)
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.install
include $(infra)/make/Makefile.clean
