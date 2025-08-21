from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, cast, overload

from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agents.exceptions import ModelBehaviorError

try:
    import litellm
except ImportError as _e:
    raise ImportError(
        "`litellm` is required to use the LitellmModel. You can install it via the optional "
        "dependency group: `pip install 'openai-agents[litellm]'`."
    ) from _e

from openai import NOT_GIVEN, AsyncStream, NotGiven
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message import (
    Annotation,
    AnnotationURLCitation,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import Response
from openai.types.responses.response_output_text import Logprob, LogprobTopLogprob

from ... import _debug
from ...agent_output import AgentOutputSchemaBase
from ...handoffs import Handoff
from ...items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ...logger import logger
from ...model_settings import ModelSettings
from ...models.chatcmpl_converter import Converter
from ...models.chatcmpl_helpers import HEADERS
from ...models.chatcmpl_stream_handler import ChatCmplStreamHandler
from ...models.fake_id import FAKE_RESPONSES_ID
from ...models.interface import Model, ModelTracing
from ...tool import Tool
from ...tracing import generation_span
from ...tracing.span_data import GenerationSpanData
from ...tracing.spans import Span
from ...usage import Usage


# <<< MODIFICATION START: 2. 更新 InternalChatCompletionMessage 的类型注解 >>>
class InternalChatCompletionMessage(ChatCompletionMessage):
    """
    An internal subclass to carry reasoning_content without modifying the original model.
    """
    reasoning_content: str
    logprobs: list[Logprob] | None
    # 告诉它 tool_calls 列表将包含我们上面的新类型
    tool_calls: list[InternalChatCompletionMessageToolCall] | None
# <<< MODIFICATION END >>>


# <<< MODIFICATION START: 1. 定义新的内部工具调用类 >>>
class InternalChatCompletionMessageToolCall(ChatCompletionMessageToolCall):
    """
    一个内部子类，用于携带工具调用特有的logprobs。
    """
    logprobs: list[Logprob] | None = None
# <<< MODIFICATION END >>>


class LitellmModel(Model):
    """This class enables using any model via LiteLLM. LiteLLM allows you to acess OpenAPI,
    Anthropic, Gemini, Mistral, and many other models.
    See supported models here: [litellm models](https://docs.litellm.ai/docs/providers).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt: Any | None = None,
    ) -> ModelResponse:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=False,
                prompt=prompt,
            )

            assert isinstance(response.choices[0], litellm.types.utils.Choices)

            if _debug.DONT_LOG_MODEL_DATA:
                logger.debug("Received model response")
            else:
                logger.debug(
                    f"""LLM resp:\n{
                        json.dumps(
                            response.choices[0].message.model_dump(), indent=2, ensure_ascii=False
                        )
                    }\n"""
                )

            if hasattr(response, "usage"):
                response_usage = response.usage
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=response_usage.prompt_tokens,
                        output_tokens=response_usage.completion_tokens,
                        total_tokens=response_usage.total_tokens,
                        input_tokens_details=InputTokensDetails(
                            cached_tokens=getattr(
                                response_usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        ),
                        output_tokens_details=OutputTokensDetails(
                            reasoning_tokens=getattr(
                                response_usage.completion_tokens_details, "reasoning_tokens", 0
                            )
                            or 0
                        ),
                    )
                    if response.usage
                    else Usage()
                )
            else:
                usage = Usage()
                logger.warning("No usage information returned from Litellm")

            if tracing.include_data():
                span_generation.span_data.output = [response.choices[0].message.model_dump()]
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

            items = Converter.message_to_output_items(
                LitellmConverter.convert_message_to_openai(
                    response.choices[0].message, 
                    # response.choices[0].logprobs
                    logprobs=getattr(response.choices[0], "logprobs", None)
                )
            )

            return ModelResponse(
                output=items,
                usage=usage,
                response_id=None,
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response, stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span_generation,
                tracing,
                stream=True,
                prompt=prompt,
            )

            final_response: Response | None = None
            async for chunk in ChatCmplStreamHandler.handle_stream(response, stream):
                yield chunk

                if chunk.type == "response.completed":
                    final_response = chunk.response

            if tracing.include_data() and final_response:
                span_generation.span_data.output = [final_response.model_dump()]

            if final_response and final_response.usage:
                span_generation.span_data.usage = {
                    "input_tokens": final_response.usage.input_tokens,
                    "output_tokens": final_response.usage.output_tokens,
                }

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[True],
        prompt: Any | None = None,
    ) -> tuple[Response, AsyncStream[ChatCompletionChunk]]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: Literal[False],
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        converted_messages = Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else None
        )
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                f"Calling Litellm model: {self.model}\n"
                f"{json.dumps(converted_messages, indent=2, ensure_ascii=False)}\n"
                f"Tools:\n{json.dumps(converted_tools, indent=2, ensure_ascii=False)}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        reasoning_effort = model_settings.reasoning.effort if model_settings.reasoning else None

        stream_options = None
        if stream and model_settings.include_usage is not None:
            stream_options = {"include_usage": model_settings.include_usage}

        extra_kwargs = {}
        if model_settings.extra_query:
            extra_kwargs["extra_query"] = model_settings.extra_query
        if model_settings.metadata:
            extra_kwargs["metadata"] = model_settings.metadata
        if model_settings.extra_body and isinstance(model_settings.extra_body, dict):
            extra_kwargs.update(model_settings.extra_body)

        # Add kwargs from model_settings.extra_args, filtering out None values
        if model_settings.extra_args:
            extra_kwargs.update(model_settings.extra_args)

        ret = await litellm.acompletion(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools or None,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            frequency_penalty=model_settings.frequency_penalty,
            presence_penalty=model_settings.presence_penalty,
            max_tokens=model_settings.max_tokens,
            tool_choice=self._remove_not_given(tool_choice),
            response_format=self._remove_not_given(response_format),
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            stream_options=stream_options,
            reasoning_effort=reasoning_effort,
            extra_headers={**HEADERS, **(model_settings.extra_headers or {})},
            api_key=self.api_key,
            base_url=self.base_url,
            **extra_kwargs,
        )

        if isinstance(ret, litellm.types.utils.ModelResponse):
            return ret

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
            if tool_choice != NOT_GIVEN
            else "auto",
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, ret

    def _remove_not_given(self, value: Any) -> Any:
        if isinstance(value, NotGiven):
            return None
        return value


class LitellmConverter:
    @classmethod
    def convert_message_to_openai(
        cls,
        message: litellm.types.utils.Message,
        logprobs: litellm.types.utils.ChoiceLogprobs | None = None,
    ) -> ChatCompletionMessage:
        if message.role != "assistant":
            raise ModelBehaviorError(f"Unsupported role: {message.role}")

        # <<< MODIFICATION START: 3. 修改此方法的核心逻辑 >>>
        
        # 首先，一次性转换所有logprobs
        full_openai_logprobs = cls.convert_logprobs_to_openai(logprobs) if logprobs else None

        tool_calls_with_logprobs: list[InternalChatCompletionMessageToolCall] | None = None
        content_logprobs: list[Logprob] | None = None

        if message.tool_calls:
            # 规则：如果存在工具调用，我们假设所有logprobs都属于它们。
            # 为简单起见，我们将所有logprobs赋给第一个工具调用。
            converted_tool_calls = []
            for i, tool in enumerate(message.tool_calls):
                # 将完整的logprobs列表赋给第一个工具调用，后续的为空。
                # 这是一个合理的简化，因为通常一次只会生成一个需要分析的工具调用。
                current_tool_logprobs = full_openai_logprobs if i == 0 else None
                converted_tool_calls.append(
                    cls.convert_tool_call_to_openai(tool, current_tool_logprobs)
                )
            tool_calls_with_logprobs = converted_tool_calls
        else:
            # 规则：如果没有工具调用，logprobs才属于文本内容。
            content_logprobs = full_openai_logprobs

        # provider_specific_fields, refusal, reasoning_content 的代码保持不变...
        provider_specific_fields = message.get("provider_specific_fields", None)
        refusal = (
            provider_specific_fields.get("refusal", None) if provider_specific_fields else None
        )
        reasoning_content = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning_content = message.reasoning_content

        # 使用我们新定义的内部类来创建消息对象
        return InternalChatCompletionMessage(
            content=message.content,
            refusal=refusal,
            role="assistant",
            annotations=cls.convert_annotations_to_openai(message),
            audio=message.get("audio", None),
            # 传入带有logprobs的工具调用列表
            tool_calls=tool_calls_with_logprobs,
            reasoning_content=reasoning_content,
            # 传入属于文本内容的logprobs
            logprobs=content_logprobs,
        )
        # <<< MODIFICATION END >>>

    @classmethod
    def convert_logprobs_to_openai(
        cls, logprobs: litellm.types.utils.ChoiceLogprobs | None # <-- 允许None
    ) -> list[Logprob] | None: # <-- 允许返回None
        # <<< MODIFICATION START: 4. 使方法更健壮 >>>
        if not logprobs or not logprobs.content:
            return None
        # <<< MODIFICATION END >>>
        return [
            Logprob(
                token=logprob.token,
                logprob=logprob.logprob,
                bytes=cast(list[int], logprob.bytes),
                top_logprobs=[
                    LogprobTopLogprob(
                        token=top_logprob.token,
                        logprob=top_logprob.logprob,
                        bytes=cast(list[int], top_logprob.bytes),
                    )
                    for top_logprob in logprob.top_logprobs
                ],
            )
            for logprob in logprobs.content
        ]

    @classmethod
    def convert_annotations_to_openai(
        cls, message: litellm.types.utils.Message
    ) -> list[Annotation] | None:
        annotations: list[litellm.types.llms.openai.ChatCompletionAnnotation] | None = message.get(
            "annotations", None
        )
        if not annotations:
            return None

        return [
            Annotation(
                type="url_citation",
                url_citation=AnnotationURLCitation(
                    start_index=annotation["url_citation"]["start_index"],
                    end_index=annotation["url_citation"]["end_index"],
                    url=annotation["url_citation"]["url"],
                    title=annotation["url_citation"]["title"],
                ),
            )
            for annotation in annotations
        ]

    @classmethod
    def convert_tool_call_to_openai(
        cls, 
        tool_call: litellm.types.utils.ChatCompletionMessageToolCall,
        # <<< MODIFICATION START: 5. 接收分配好的logprobs >>>
        assigned_logprobs: list[Logprob] | None
        # <<< MODIFICATION END >>>
    ) -> InternalChatCompletionMessageToolCall: # <<< 6. 更改返回类型
        # <<< MODIFICATION START: 7. 使用新的内部类实例化 >>>
        return InternalChatCompletionMessageToolCall( # <-- 使用新类
            id=tool_call.id,
            type="function",
            function=Function(
                name=tool_call.function.name or "", arguments=tool_call.function.arguments
            ),
            logprobs=assigned_logprobs, # <-- 传入logprobs
        )
        # <<< MODIFICATION END >>>