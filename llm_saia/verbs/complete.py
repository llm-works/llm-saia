"""COMPLETE verb: Execute a task with tool calling and completion confirmation."""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from llm_saia.core.backend import AgentResponse, Message, ToolCall
from llm_saia.core.config import Config, RunConfig
from llm_saia.core.types import TaskResult
from llm_saia.core.verb import Verb

# Default run config for complete (unlimited iterations)
DEFAULT_COMPLETE_RUN = RunConfig(max_iterations=0)


class Complete(Verb):
    """Execute a task with tool calling and completion confirmation."""

    async def __call__(
        self,
        task: str,
        on_iteration: Callable[[int, AgentResponse], Awaitable[None]] | None = None,
    ) -> TaskResult:
        """Execute a task using tools until completion or limit reached."""
        if not self._has_tools():
            raise ValueError("Complete requires tools and executor to be configured.")

        config = self._config.run or DEFAULT_COMPLETE_RUN
        messages: list[Message] = [Message(role="user", content=task)]
        start_time, iteration, total_tokens, last_content = time.monotonic(), 0, 0, ""
        self._log_loop_start(config)

        while not self._should_stop(config, iteration, start_time, total_tokens):
            response, tokens = await self._run_iteration(messages, config)
            total_tokens, last_content = total_tokens + tokens, response.content
            self._log_response(response, iteration, total_tokens)
            self._check_tool_support(response)
            if on_iteration:
                await on_iteration(iteration, response)

            result = await self._try_complete(task, response, messages, iteration)
            if result:
                self._log_loop_complete(
                    iteration, start_time, total_tokens, self._result_preview(result)
                )
                return result
            iteration += 1

        self._log_limit_reached(config, iteration, start_time, total_tokens)
        return TaskResult(False, last_content, iteration, messages)

    async def _try_complete(
        self, task: str, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if task completed via terminal tool or confirmation."""
        terminal_result = await self._check_terminal_tool(task, response, messages, iteration)
        if terminal_result is False:
            # Terminal tool detected but awaiting confirmation - continue loop
            return None
        if terminal_result is not None:
            return terminal_result
        return await self._handle_response(task, response, messages, iteration)

    def _result_preview(self, result: TaskResult) -> str:
        """Get preview content from TaskResult for logging."""
        if result.output:
            return result.output
        if result.terminal_data:
            return self._safe_json_dumps(result.terminal_data)
        return ""

    def _safe_json_dumps(self, data: object, indent: int | None = None) -> str:
        """Serialize data to JSON, falling back to str() for non-serializable objects."""
        try:
            return json.dumps(data, indent=indent)
        except (TypeError, ValueError):
            return str(data)

    async def _check_terminal_tool(
        self, task: str, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | Literal[False] | None:
        """Check if terminal tool was called.

        Returns:
            TaskResult: Terminal tool confirmed, task complete
            False: Terminal tool detected, confirmation injected, continue loop
            None: No terminal tool, proceed with normal handling
        """
        terminal_tool = self._config.terminal_tool
        if not terminal_tool or not response.tool_calls:
            return None

        terminal_call = next((tc for tc in response.tool_calls if tc.name == terminal_tool), None)
        if not terminal_call:
            return None

        messages.append(self._to_message(response))

        # Check if this is a confirmed terminal call (called twice in a row)
        if self._is_terminal_confirmed(messages, terminal_tool):
            return self._handle_confirmed_terminal(
                task, terminal_tool, terminal_call, response.content, messages, iteration
            )

        # Execute any non-terminal tools before asking for confirmation
        # This ensures the LLM has results from all tools it called
        non_terminal_calls = [tc for tc in response.tool_calls if tc.name != terminal_tool]
        if non_terminal_calls:
            await self._execute_tools(non_terminal_calls, messages)

        # First terminal call - ask for confirmation
        self._inject_terminal_confirmation(task, terminal_tool, terminal_call, messages)
        return False

    # Marker in pushback messages that invalidates a pending confirmation
    _CONTRADICTION_MARKER = "contradictory signals"

    def _is_terminal_confirmed(self, messages: list[Message], terminal_tool: str) -> bool:
        """Check if this is a second terminal tool call (confirmation)."""
        # Look for a recent confirmation prompt in messages
        confirm_marker = f"call `{terminal_tool}` again to confirm"
        for msg in reversed(messages[:-1]):  # Exclude the message we just added
            # If we hit a contradiction pushback, the confirmation was invalidated
            if msg.role == "tool_result" and self._CONTRADICTION_MARKER in msg.content:
                return False
            if msg.role == "user" and confirm_marker in msg.content:
                return True
            # Only look back to the previous user message
            if msg.role == "user":
                break
        return False

    def _check_contradiction(
        self, confirm_content: str, terminal_call: ToolCall, messages: list[Message]
    ) -> bool:
        """Check for contradictory continuation signals. Returns True if pushed back."""
        if not self._has_continuation_signal(confirm_content):
            return False
        contradiction_msg = (
            "Your response contains contradictory signals - you confirmed completion "
            "but also indicated you want to continue. Please either continue working "
            "using the available tools, or call the terminal tool with a clear final answer."
        )
        messages.append(
            Message(role="tool_result", content=contradiction_msg, tool_call_id=terminal_call.id)
        )
        return True

    def _handle_confirmed_terminal(
        self,
        task: str,
        terminal_tool: str,
        terminal_call: ToolCall,
        confirm_content: str,
        messages: list[Message],
        iteration: int,
    ) -> TaskResult | Literal[False]:
        """Handle a confirmed terminal tool call."""
        # Check for contradiction: LLM says "confirmed" but also has continuation signals
        if self._check_contradiction(confirm_content, terminal_call, messages):
            return False

        # Check for failure indicators - if LLM says "stuck", push back
        if self._is_failure_status(terminal_call.arguments):
            max_retries = (self._config.run or DEFAULT_COMPLETE_RUN).max_retries
            if self._count_failure_retries(messages) < max_retries:
                self._inject_retry_prompt(task, terminal_call.arguments, messages)
                return False
            # Max retries exceeded - accept failure

        original_content = self._get_original_terminal_content(messages, terminal_tool)
        return TaskResult(
            completed=not self._is_failure_status(terminal_call.arguments),
            output=original_content,
            iterations=iteration + 1,
            history=messages,
            terminal_data=terminal_call.arguments,
            terminal_tool=terminal_tool,
        )

    def _get_original_terminal_content(self, messages: list[Message], terminal_tool: str) -> str:
        """Get content from the original terminal tool call (before confirmation).

        Searches backwards for the assistant message that precedes the confirmation prompt.
        This handles cases where non-terminal tools added extra messages.
        """
        confirm_marker = f"call `{terminal_tool}` again to confirm"
        found_confirm = False
        for msg in reversed(messages[:-1]):  # Exclude the confirmation response we just added
            if msg.role == "user" and confirm_marker in msg.content:
                found_confirm = True
                continue
            if found_confirm and msg.role == "assistant":
                return msg.content
        return ""

    def _inject_terminal_confirmation(
        self, task: str, terminal_tool: str, terminal_call: ToolCall, messages: list[Message]
    ) -> None:
        """Inject confirmation prompt for terminal tool call."""
        # Add tool result acknowledging the call
        messages.append(
            Message(
                role="tool_result",
                content="Received. Please confirm this is your final response.",
                tool_call_id=terminal_call.id,
            )
        )
        # Add confirmation prompt
        data_preview = self._safe_json_dumps(terminal_call.arguments, indent=2)
        prompt = (
            f"You called `{terminal_tool}` to signal completion.\n\n"
            f"**Original task:** {task}\n\n"
            f"**Your response:**\n```json\n{data_preview}\n```\n\n"
            f"Is this your final response to the task?\n"
            f"- If YES, call `{terminal_tool}` again to confirm.\n"
            f"- If NO, continue working using the available tools."
        )
        messages.append(Message(role="user", content=prompt))

    # Statuses that indicate the LLM gave up without completing
    _FAILURE_STATUSES = frozenset(
        {
            "stuck",
            "failed",
            "error",
            "incomplete",
            "blocked",
            "unable",
            "cannot",
            "impossible",
            "give_up",
            "giving_up",
            "abort",
            "aborted",
        }
    )

    def _is_failure_status(self, terminal_data: dict[str, Any]) -> bool:
        """Check if terminal data indicates a failure rather than success."""
        status = terminal_data.get("status", "")
        if isinstance(status, str):
            return status.lower() in self._FAILURE_STATUSES
        return False

    # Marker for retry prompts so we can count them
    _RETRY_MARKER = "[SAIA_RETRY]"

    def _inject_retry_prompt(
        self, task: str, terminal_data: dict[str, Any], messages: list[Message]
    ) -> None:
        """Inject prompt telling LLM to keep trying after failure status."""
        status = terminal_data.get("status", "unknown")
        conclusion = terminal_data.get("conclusion", "")
        prompt = (
            f"{self._RETRY_MARKER} You indicated status '{status}' but the task is not complete.\n"
            f"\n**Original task:** {task}\n\n"
            f"**Your conclusion:** {conclusion}\n\n"
            f"Please continue working on the task using the available tools. "
            f"Do not give up - try a different approach if needed."
        )
        messages.append(Message(role="user", content=prompt))

    def _count_failure_retries(self, messages: list[Message]) -> int:
        """Count how many retry prompts have been injected."""
        return sum(
            1 for msg in messages if msg.role == "user" and self._RETRY_MARKER in msg.content
        )

    async def _run_iteration(
        self, messages: list[Message], config: RunConfig
    ) -> tuple[AgentResponse, int]:
        """Run one LLM iteration and return response with token count."""
        max_tokens = config.max_call_tokens if config.max_call_tokens > 0 else None
        response = await self._chat(messages, max_tokens)
        return response, response.input_tokens + response.output_tokens

    async def _handle_response(
        self, task: str, response: AgentResponse, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Handle LLM response - execute tools or check completion."""
        messages.append(self._to_message(response))

        if response.tool_calls:
            await self._execute_tools(response.tool_calls, messages)
            return None

        return await self._check_completion(task, response.content, messages, iteration)

    def _single_call_config(self) -> Config:
        """Create config for single-call verbs (no tools/looping)."""
        return Config(
            backend=self._config.backend,
            tools=[],
            executor=None,
            system=self._config.system,
            run=None,
            terminal_tool=None,
            lg=self._config.lg,
            warn_tool_support=self._config.warn_tool_support,
        )

    # Phrases that indicate the LLM wants to continue but didn't use tools
    _CONTINUATION_SIGNALS = (
        # Intent to act
        "let's proceed",
        "let me ",
        "i will use",
        "i'll use",
        "next, i will",
        "next i will",
        "now i'll",
        "now i will",
        "let's use",
        "i'm going to use",
        "i am going to use",
        # Asking for permission to continue
        "would you like to proceed",
        "would you like me to",
        "shall i ",
        "should i ",
        "do you want me to",
        "do you want to proceed",
        "want me to continue",
        # Intent to explore/read (often when tool access is lost)
        "let's read",
        "let's examine",
        "let's look at",
        "let's check",
        "let's explore",
        "let's review",
        "let's see",
        "let's open",
    )

    # Patterns that look like tool invocations written in text (when tool access is lost)
    _TEXT_TOOL_PATTERNS = (
        "read_file",
        "shell ",
        "execute(",
        "run_command",
        "search_files",
        "list_files",
    )

    def _has_continuation_signal(self, content: str) -> bool:
        """Check if content indicates the LLM wants to continue working."""
        content_lower = content.lower()
        if any(signal in content_lower for signal in self._CONTINUATION_SIGNALS):
            return True
        # Check for tool invocations written as text (usually in code blocks)
        if any(pattern in content_lower for pattern in self._TEXT_TOOL_PATTERNS):
            return True
        return False

    async def _check_completion(
        self, task: str, content: str, messages: list[Message], iteration: int
    ) -> TaskResult | None:
        """Check if task is complete. Returns TaskResult if done, None to continue."""
        # Fast path: if content has continuation signals, don't bother with Confirm
        if self._has_continuation_signal(content):
            wrap_up = (
                "You indicated you want to continue but didn't use any tools. "
                "Please use the available tools to proceed, or call the terminal tool "
                "if you have completed the task."
            )
            messages.append(Message(role="user", content=wrap_up))
            return None

        from llm_saia.verbs.confirm import Confirm

        confirm = Confirm(self._single_call_config())
        confirmation = await confirm(
            claim="the task is complete based on the agent's response",
            context=f"Task: {task}\n\nAgent's response: {content}",
        )

        if confirmation.confirmed:
            return TaskResult(
                completed=True, output=content, iterations=iteration + 1, history=messages
            )

        wrap_up = (
            f"The task is not yet complete. Reason: {confirmation.reason}\n"
            "Please continue working on the task or use the available tools."
        )
        messages.append(Message(role="user", content=wrap_up))
        return None
