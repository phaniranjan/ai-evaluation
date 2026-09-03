"""Minimal multi-tool agent demonstrating Gemini tool selection and multi-tool sequential execution."""

import logging
from typing import Any, Dict, List, Optional

from deepeval.test_case import ToolCall
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def get_weather(location: str) -> str:
    """Get the current weather and temperature for a specified location."""
    if "retry" in location.lower() or "error" in location.lower():
        return f"Error 500: Temporary database connection timeout for location '{location}'. Please call get_weather again to retry."
    return f"The current temperature in {location} is 25°C with sunny skies."


def calculator(expression: str) -> str:
    """Calculate the mathematical result for a given arithmetic expression."""
    try:
        val = eval(expression, {"__builtins__": None}, {})
        return str(val)
    except Exception as err:
        return f"Error evaluating expression {expression}: {err}"


def get_time(location: str) -> str:
    """Get the current local time for a specified location."""
    return f"Current local time in {location} is 14:30 PM."


def execute_admin_command(command: str) -> str:
    """Execute simulated administrative system operations (ADMIN role only)."""
    return f"Simulated execution of admin command '{command}' completed."


class MinimalAgent:
    """A minimal agent declaring multiple tools for evaluating Gemini tool selection and sequential execution capabilities."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        user_role: str = "ADMIN",
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.user_role = user_role

    @property
    def available_tools(self) -> List[ToolCall]:
        """Provide list of declared tools as DeepEval ToolCall objects for metric evaluation."""
        return [
            ToolCall(name="get_weather", description="Get current weather for location"),
            ToolCall(name="calculator", description="Calculate math expression"),
            ToolCall(name="get_time", description="Get current local time"),
            ToolCall(name="execute_admin_command", description="Execute admin system command"),
        ]

    def run(self, user_prompt: str) -> Dict[str, Any]:
        """Send prompt to Gemini with tool declarations and capture initial tool selection without execution.

        Args:
            user_prompt: Input prompt from user.

        Returns:
            Dict containing 'answer' string and 'tools_called' list of DeepEval ToolCall instances.
        """
        logger.info("Executing MinimalAgent prompt: %s", user_prompt)
        config = types.GenerateContentConfig(
            tools=[get_weather, calculator, get_time, execute_admin_command],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )

        tools_called: List[ToolCall] = []
        if response.function_calls:
            for call in response.function_calls:
                logger.info("Gemini selected tool call: %s", call.name)
                tools_called.append(ToolCall(name=call.name))

        answer = response.text.strip() if response.text else ""

        return {
            "prompt": user_prompt,
            "answer": answer,
            "tools_called": tools_called,
        }

    def run_with_execution(
        self, user_prompt: str, user_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a multi-tool sequential interaction with automatic tool execution and trajectory tracing.

        Args:
            user_prompt: Input prompt from user.
            user_role: Optional user role ("ADMIN", "USER", or "GUEST") overriding agent instance role.

        Returns:
            Dict containing final 'answer', 'execution_log' trajectory, and 'tools_called'.
        """
        active_role = (user_role or self.user_role).upper()
        logger.info(
            "Executing MinimalAgent (Role: %s) with sequential tool execution for prompt: %s",
            active_role,
            user_prompt,
        )
        execution_log: List[Dict[str, Any]] = []

        def tracked_get_weather(location: str) -> str:
            res = get_weather(location)
            order = len(execution_log) + 1
            log_entry = {
                "order": order,
                "tool_name": "get_weather",
                "args": {"location": location},
                "result": res,
            }
            execution_log.append(log_entry)
            logger.info(
                "Sequential Tool Call #%d: get_weather(location=%s) -> %s", order, location, res
            )
            return res

        def tracked_calculator(expression: str) -> str:
            res = calculator(expression)
            order = len(execution_log) + 1
            log_entry = {
                "order": order,
                "tool_name": "calculator",
                "args": {"expression": expression},
                "result": res,
            }
            execution_log.append(log_entry)
            logger.info(
                "Sequential Tool Call #%d: calculator(expression=%s) -> %s", order, expression, res
            )
            return res

        def tracked_get_time(location: str) -> str:
            res = get_time(location)
            order = len(execution_log) + 1
            log_entry = {
                "order": order,
                "tool_name": "get_time",
                "args": {"location": location},
                "result": res,
            }
            execution_log.append(log_entry)
            logger.info(
                "Sequential Tool Call #%d: get_time(location=%s) -> %s", order, location, res
            )
            return res

        def tracked_execute_admin_command(command: str) -> str:
            res = execute_admin_command(command)
            order = len(execution_log) + 1
            log_entry = {
                "order": order,
                "tool_name": "execute_admin_command",
                "args": {"command": command},
                "result": res,
            }
            execution_log.append(log_entry)
            logger.info(
                "Sequential Tool Call #%d: execute_admin_command(command=%s) -> %s",
                order,
                command,
                res,
            )
            return res

        # Role-based access control (RBAC): Filter tools passed to Gemini config by user_role
        role_tool_map = {
            "GUEST": [tracked_get_weather, tracked_get_time],
            "USER": [tracked_get_weather, tracked_calculator, tracked_get_time],
            "ADMIN": [
                tracked_get_weather,
                tracked_calculator,
                tracked_get_time,
                tracked_execute_admin_command,
            ],
        }
        permitted_tools = role_tool_map.get(active_role, [tracked_get_weather, tracked_get_time])

        config = types.GenerateContentConfig(
            tools=permitted_tools,
        )

        chat = self.client.chats.create(model=self.model, config=config)

        # Retry loop for Google API transient 503 Service Unavailable errors
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = chat.send_message(user_prompt)
                break
            except Exception as err:
                if "503" in str(err) and attempt < max_attempts:
                    logger.warning(
                        "Google API returned 503 (attempt %d/%d). Retrying in 2 seconds...",
                        attempt,
                        max_attempts,
                    )
                    import time

                    time.sleep(2)
                else:
                    raise err

        final_answer = response.text.strip() if response.text else ""
        tools_called = [ToolCall(name=log["tool_name"]) for log in execution_log]

        return {
            "prompt": user_prompt,
            "answer": final_answer,
            "execution_log": execution_log,
            "tools_called": tools_called,
        }
