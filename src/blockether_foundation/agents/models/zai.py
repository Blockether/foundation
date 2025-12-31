from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from os import getenv
from typing import Any, override

import httpx
from agno.exceptions import ModelProviderError
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput
from agno.run.team import TeamRunOutput
from agno.utils.log import log_debug, log_error  # type: ignore
from pydantic import BaseModel


@dataclass
class Zhipu(OpenAILike):
    """Zhipu AI model compatible with OpenAI API format with enhanced features."""

    id: str = "glm-4.7"
    name: str = "Zhipu"
    provider: str = "Zhipu"

    api_key: str | None = None
    base_url: str | httpx.URL | None = None

    # Thinking mode configuration
    enable_thinking: bool = False

    # Supports structured outputs
    supports_native_structured_outputs: bool = False

    # Coding plan configuration
    enable_coding_plan: bool = False

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        # Set base_url if not provided
        if self.base_url is None:
            if self.enable_coding_plan:
                self.base_url = "https://api.z.ai/api/coding/paas/v4"
            else:
                self.base_url = "https://open.bigmodel.cn/api/paas/v4"

        # Set api_key if not provided (check both Z_AI_API_KEY and ZHIPU_API_KEY)
        if self.api_key is None:
            self.api_key = getenv("Z_AI_API_KEY") or getenv("ZHIPU_API_KEY")

    def _configure_thinking_params(self, enabled: bool) -> dict[str, Any]:
        """Configure thinking parameters"""
        return {"type": "enabled"} if enabled else {"type": "disabled"}

    @override
    def _get_client_params(self) -> dict[str, Any]:
        """Get client parameters for OpenAI client."""
        if not self.api_key:
            self.api_key = getenv("Z_AI_API_KEY") or getenv("ZHIPU_API_KEY")
            if not self.api_key:
                raise ModelProviderError(
                    message="Z_AI_API_KEY or ZHIPU_API_KEY not set. Please set one of these environment variables.",
                    model_name=self.name,
                    model_id=self.id,
                )

        # Define base client params with proper type annotation
        base_params: dict[str, Any] = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        # Create client_params dict with non-None values
        client_params: dict[str, Any] = {k: v for k, v in base_params.items() if v is not None}

        # Add additional client params if provided
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    @override
    def get_request_params(
        self,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_response: RunOutput | TeamRunOutput | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get request parameters with thinking mode support.
        """
        # Get base parameters (includes response_format handling from OpenAIChat)
        params = super().get_request_params(
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            **kwargs,
        )

        if "extra_body" not in params:
            params["extra_body"] = {}
        params["extra_body"]["thinking"] = self._configure_thinking_params(self.enable_thinking)

        return params

    @override
    def invoke(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_response: RunOutput | TeamRunOutput | None = None,
        compress_tool_results: bool = False,
    ) -> ModelResponse:
        """Enhanced non-streaming call with performance optimization."""
        # Ensure API key is set before calling parent invoke and log error early
        try:
            self._get_client_params()
        except ModelProviderError as e:
            log_error(e.message)
            raise e

        # Use agno's metrics system
        if run_response and run_response.metrics:
            run_response.metrics.set_time_to_first_token()

        assistant_message.metrics.start_timer()

        # Use agno's standard logging system - basic logging handled by base.py
        if self.request_params:
            log_debug(
                f"Calling {self.provider} with request parameters: {self.request_params}",
                log_level=2,
            )

        # Execute request
        response = super().invoke(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

        assistant_message.metrics.stop_timer()
        return response

    @override
    def invoke_stream(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_response: RunOutput | TeamRunOutput | None = None,
        compress_tool_results: bool = False,
    ) -> Iterator[ModelResponse]:
        """Enhanced streaming call."""
        # Ensure API key is set before calling parent invoke and log error early
        try:
            self._get_client_params()
        except ModelProviderError as e:
            log_error(e.message)
            raise e

        # Use agno's metrics system
        if run_response and run_response.metrics:
            run_response.metrics.set_time_to_first_token()

        assistant_message.metrics.start_timer()

        # Use agno's standard logging system - basic logging handled by base.py
        if self.request_params:
            log_debug(
                f"Calling {self.provider} with request parameters: {self.request_params}",
                log_level=2,
            )

        # Execute streaming request
        response_stream = super().invoke_stream(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

        # Handle streaming response
        yield from response_stream

        assistant_message.metrics.stop_timer()

    @override
    async def ainvoke(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_response: RunOutput | TeamRunOutput | None = None,
        compress_tool_results: bool = False,
    ) -> ModelResponse:
        """Enhanced async non-streaming call."""
        # Ensure API key is set before calling parent invoke and log error early
        try:
            self._get_client_params()
        except ModelProviderError as e:
            log_error(e.message)
            raise e

        # Use agno's metrics system
        if run_response and run_response.metrics:
            run_response.metrics.set_time_to_first_token()

        assistant_message.metrics.start_timer()

        # Use agno's standard logging system - basic logging handled by base.py
        if self.request_params:
            log_debug(
                f"Calling {self.provider} with request parameters: {self.request_params}",
                log_level=2,
            )

        # Execute async request
        response = await super().ainvoke(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

        assistant_message.metrics.stop_timer()
        return response

    @override
    async def ainvoke_stream(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_response: RunOutput | TeamRunOutput | None = None,
        compress_tool_results: bool = False,
    ) -> AsyncIterator[ModelResponse]:
        """Enhanced async streaming call."""
        # Ensure API key is set before calling parent invoke and log error early
        try:
            self._get_client_params()
        except ModelProviderError as e:
            log_error(e.message)
            raise e

        # Use agno's metrics system
        if run_response and run_response.metrics:
            run_response.metrics.set_time_to_first_token()

        assistant_message.metrics.start_timer()

        # Use agno's standard logging system - basic logging handled by base.py
        if self.request_params:
            log_debug(
                f"Calling {self.provider} with request parameters: {self.request_params}",
                log_level=2,
            )

        # Execute async streaming request
        response_stream = super().ainvoke_stream(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

        # Handle streaming response
        async for response in response_stream:
            yield response

        assistant_message.metrics.stop_timer()
