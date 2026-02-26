#!/usr/bin/env python3
"""Agent with tool loop: complete() iterates until done

Requires a model with robust tool calling support. Small or base models often fail
to call tools correctly. Options:
  - Anthropic: claude-haiku or higher (recommended)
  - OpenAI: gpt-4o-mini or higher
  - Local: 14B+ instruction-tuned models with tool support

Recommended usage: LLM_BACKEND=anthropic ./agent.py
Requires ANTHROPIC_API_KEY environment variable.

Example output:
    [task] Read the Python files in ./examples and briefly describe what each one does

    [0] list_files(path=./examples)
    [1] read_file(path=./examples/__init__.py)
    [1] read_file(path=./examples/agent.py)
    [1] read_file(path=./examples/investigate.py)
    ...
    [2] done(answer=## Python Files in ./examples ...)

    [done] iterations: 3

    **`__init__.py`** - Shared utilities module for SAIA examples
    **`agent.py`** - Demonstrates an agentic tool loop with the Complete verb
    **`investigate.py`** - Implements a fact-checking workflow: verify → critique → refine
    ...
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import Colors as C
from examples import get_backend
from llm_saia import SAIA, ToolDef

# Define tools the agent can use
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
        name="list_files",
        description="List files in a directory",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path"}},
            "required": ["path"],
        },
    ),
    ToolDef(
        name="done",
        description=(
            "Call this when you have the final answer. "
            "The answer field should contain your complete response."
        ),
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your complete final answer to the task",
                }
            },
            "required": ["answer"],
        },
    ),
]


async def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool call."""
    match name:
        case "read_file":
            path = Path(args["path"])
            if not path.exists():
                return f"Error: {path} not found"
            return path.read_text()[:2000]  # Limit for demo
        case "list_files":
            path = Path(args["path"])
            if not path.is_dir():
                return f"Error: {path} is not a directory"
            files = [p.name for p in sorted(path.iterdir()) if not p.name.startswith(".")]
            return "\n".join(files[:20])  # Limit for demo
        case _:
            return f"Unknown tool: {name}"


async def main() -> None:
    async with get_backend() as backend:
        saia = (
            SAIA.builder()
            .backend(backend)
            .tools(TOOLS, execute_tool)
            .terminal("done", output_field="answer")
            .system("You are a helpful assistant. Use the provided tools to complete tasks.")
            .max_iterations(15)
            .build()
        )

        examples_dir = Path(__file__).parent
        task = f"Read the Python files in {examples_dir} and briefly describe what each one does"
        print(f"{C.CYAN}[task]{C.RESET} {task}\n")

        # Track iterations
        async def on_iteration(i: int, response: Any) -> None:
            if response.tool_calls:
                for tc in response.tool_calls:
                    args = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
                    print(f"{C.YELLOW}[{i}]{C.RESET} {tc.name}({args})")
            elif response.content:
                print(f"{C.YELLOW}[{i}]{C.RESET} thinking...")

        result = await saia.complete(task, on_iteration=on_iteration)

        print(f"\n{C.GREEN}[done]{C.RESET} iterations: {result.iterations}")
        print(f"\n{result.output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n{C.RED}[error]{C.RESET} {type(e).__name__}: {e}")
