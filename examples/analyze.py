#!/usr/bin/env python3
"""Example: Use the Complete verb with tools to analyze its own source code.

Demonstrates the agentic tool loop with tracing — the LLM reads files, reasons
about them, and calls done when finished. Every LLM call is traced.

Recommended usage: LLM_BACKEND=anthropic ./analyze.py
Requires ANTHROPIC_API_KEY environment variable.

Usage:
    ./examples/analyze.py           # normal output
    ./examples/analyze.py --trace   # with compact trace
    ./examples/analyze.py --trace --full  # with full JSON trace

Example output:
    [task] Analyze ./examples/analyze.py and summarize what it does

    [0] read_file(path=./examples/analyze.py)
    [1] done(summary=This file is a Python example demonstrating...)

    [done] iterations: 2

    This file is a Python example demonstrating an agentic tool loop using the
    SAIA framework. It shows how to define tools (read_file, done), configure
    terminal conditions, and trace LLM interactions...
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import Colors as C
from examples import get_backend, print_trace_compact, print_trace_full
from llm_saia import SAIA, ToolDef

# Simple tools for file analysis
TOOLS = [
    ToolDef(
        name="read_file",
        description="Read contents of a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    ),
    ToolDef(
        name="done",
        description="Call when analysis is complete",
        parameters={
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Analysis summary"}},
            "required": ["summary"],
        },
    ),
]


async def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool call."""
    if name == "read_file":
        path = Path(args["path"])
        if not path.exists():
            return f"Error: {path} not found"
        try:
            return path.read_text()[:4000]
        except (PermissionError, UnicodeDecodeError, OSError) as e:
            return f"Error reading {path}: {e}"
    return f"Unknown tool: {name}"


async def main() -> None:  # cq: exempt
    parser = argparse.ArgumentParser(description="Analyze this script with tracing")
    parser.add_argument("--full", action="store_true", help="Show full JSON trace")
    parser.add_argument("--trace", action="store_true", help="Show trace output")
    args = parser.parse_args()

    # --full implies --trace
    if args.full:
        args.trace = True

    async with get_backend() as backend:
        builder = (
            SAIA.builder()
            .backend(backend)
            .tools(TOOLS, execute_tool)
            .terminal("done", output_field="summary")
            .system("You are a code analyst. Read the file, then call done with a brief summary.")
            .max_iterations(10)
        )
        if args.trace:
            trace_fn = print_trace_full if args.full else print_trace_compact
            builder = builder.tracing.callback(trace_fn)
        saia = builder.build()

        task = f"Analyze {__file__} and summarize what it does"
        print(f"{C.CYAN}[task]{C.RESET} {task}\n")

        async def on_iteration(i: int, response: Any) -> None:
            if response.tool_calls:
                for tc in response.tool_calls:
                    args_str = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
                    args_label = args_str if len(args_str) <= 60 else args_str[:60] + "..."
                    print(f"{C.YELLOW}[{i}]{C.RESET} {tc.name}({args_label})")
            elif response.content:
                print(f"{C.YELLOW}[{i}]{C.RESET} thinking...")

        result = await saia.with_request_id("analyze-001").complete(task, on_iteration=on_iteration)
        print(f"\n{C.GREEN}[done]{C.RESET} iterations: {result.iterations}")
        print(f"\n{result.output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n{C.RED}[error]{C.RESET} {type(e).__name__}: {e}")
        sys.exit(1)
