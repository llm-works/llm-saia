# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors

"""Tests for the file-backed Tracer: JSONL written to disk."""

from __future__ import annotations

import json
from pathlib import Path

from llm_saia.core.trace import LLMCall, Step, TracerFactory, VerbTrace


class TestFileTracer:
    """Tests for Tracer JSONL output to a file."""

    def test_writes_step_as_jsonl(self, tmp_path: Path) -> None:
        """Tracer serializes a Step to JSONL."""
        path = tmp_path / "trace.jsonl"
        step = Step(
            phase="attempt",
            trace_id="t1",
            verb="Ask",
            llm_call=LLMCall(call_id="c1", input_tokens=100, output_tokens=20),
        )
        with TracerFactory.file(path) as tracer:
            tracer.write(step)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["type"] == "step"
        assert data["phase"] == "attempt"
        assert data["trace_id"] == "t1"
        assert data["llm_call"]["call_id"] == "c1"

    def test_writes_verb_trace_as_jsonl(self, tmp_path: Path) -> None:
        """Tracer serializes a VerbTrace to JSONL."""
        path = tmp_path / "trace.jsonl"
        trace = VerbTrace(
            verb="Extract",
            trace_id="t1",
            steps=[
                Step(phase="attempt", llm_call=LLMCall(call_id="c1")),
            ],
        )
        with TracerFactory.file(path) as tracer:
            tracer.write(trace)

        data = json.loads(path.read_text().strip())
        assert data["type"] == "verb_trace"
        assert data["verb"] == "Extract"
        assert len(data["steps"]) == 1

    def test_writes_metadata_header(self, tmp_path: Path) -> None:
        """Metadata is written as the first line via start()."""
        path = tmp_path / "trace.jsonl"
        step = Step(phase="attempt", trace_id="t1")
        with TracerFactory.file(path) as tracer:
            tracer.start({"trace_id": "t1", "request_id": "u1"})
            tracer.write(step)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        meta = json.loads(lines[0])
        assert meta["_meta"]["trace_id"] == "t1"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """TracerFactory.file creates parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "trace.jsonl"
        step = Step(phase="attempt")
        with TracerFactory.file(path) as tracer:
            tracer.write(step)
        assert path.exists()

    def test_close_closes_owned_writer(self, tmp_path: Path) -> None:
        """Tracer with owns_writer=True closes the writer."""
        path = tmp_path / "owned.jsonl"
        tracer = TracerFactory.file(path)
        tracer.close()
        assert path.exists()
