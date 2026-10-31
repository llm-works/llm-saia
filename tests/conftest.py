# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Shared pytest configuration for all test suites."""

# Gives each test the suite of its marker or its tests/<suite>/ directory
pytest_plugins = ["appinfra.pytest_suites"]
