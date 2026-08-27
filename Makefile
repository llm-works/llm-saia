# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

infra := $(shell appinfra scripts-path)

# Configuration
INFRA_DEV_PKG_NAME := llm_saia

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
INFRA_DEV_CQ_STRICT := true

# Include SPDX header check in `make check`.
INFRA_DEV_CQ_SPDX := true

# Test coverage threshold (percentage)
INFRA_PYTEST_COVERAGE_THRESHOLD := 95

# Docstring coverage threshold (percentage)
INFRA_DEV_DOCSTRING_THRESHOLD := 90

# Include framework (config first)
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.install
include $(infra)/make/Makefile.clean
