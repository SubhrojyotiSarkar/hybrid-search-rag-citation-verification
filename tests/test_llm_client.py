"""GeminiLLMClient against a mock HTTP transport: checks the request we send and how each
kind of response is classified. No network access and no API quota used."""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from backend.generation.llm_client import GeminiLLMClient, LLMError
from backend.schemas import GeneratedAnswer

MODEL = "gemini-3.8-flash"
VALID_OUTPUT = json.dumps({
    "answer_status": "answered",
    "missing_information": "",
    "claims": [{"text": "L2 penalizes squared weights.", "source_labels": ["S1"]}],
})


def gemini_reply(text: str = VALID_OUTPUT, finish_reason: str = "STOP", block_reason: str | None = None) -> dict:
    body = {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}] if text else []},
            "finishReason": finish_reason, "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 321, "candidatesTokenCount": 45,
                          "thoughtsTokenCount": 15, "totalTokenCount": 381},
        "modelVersion": MODEL, "responseId": "resp_test",
    }
    if block_reason:
        body = {"promptFeedback": {"blockReason": block_reason}, "modelVersion": MODEL}
    return body


def make_client(status: int = 200, body: dict | None = None, captured: dict | None = None,
                error_status: str = "", error_message: str = "boom") -> GeminiLLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
        if status != 200:
            return httpx.Response(status, json={"error": {"code": status, "message": error_message, "status": error_status}})
        return httpx.Response(200, json=body or gemini_reply())

    llm = GeminiLLMClient("test-key", MODEL, "medium", 16000, 30.0)
    llm.client = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            client_args={"transport": httpx.MockTransport(handler)},
            retry_options=types.HttpRetryOptions(attempts=1),  # no retry delays in tests
        ),
    )
    return llm


def test_valid_response_is_parsed_into_the_schema():
    result = make_client().generate_structured("system", "user", GeneratedAnswer)

    assert isinstance(result.output, GeneratedAnswer)
    assert result.output.claims[0].source_labels == ["S1"]
    assert result.input_tokens == 321
    assert result.output_tokens == 45 + 15  # answer + thinking tokens
    assert result.model == MODEL


def test_request_uses_model_system_prompt_schema_and_thinking_level():
    captured: dict = {}
    make_client(captured=captured).generate_structured("SYSTEM", "USER", GeneratedAnswer, max_tokens=999)
    body = captured["body"]
    config = body["generationConfig"]

    assert f"models/{MODEL}:generateContent" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert body["contents"][0]["parts"][0]["text"] == "USER"
    assert config["responseMimeType"] == "application/json"
    assert set(config["responseJsonSchema"]["required"]) == {"answer_status", "claims", "missing_information"}
    assert config["maxOutputTokens"] == 999
    # The SDK serializes this field in snake_case; the API accepts either spelling.
    thinking = config["thinkingConfig"]
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "MEDIUM"


def test_thought_parts_are_not_treated_as_the_answer():
    body = gemini_reply()
    body["candidates"][0]["content"]["parts"].insert(0, {"text": "thinking...", "thought": True})

    result = make_client(body=body).generate_structured("s", "u", GeneratedAnswer)

    assert result.output.answer_status == "answered"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (gemini_reply(text="", finish_reason="SAFETY"), "llm_refused"),
        (gemini_reply(block_reason="SAFETY"), "llm_refused"),
        (gemini_reply(text='{"answer_st', finish_reason="MAX_TOKENS"), "llm_truncated"),
        (gemini_reply(text='{"wrong": 1}'), "llm_invalid_output"),
        (gemini_reply(text="not json at all"), "llm_invalid_output"),
        (gemini_reply(text=""), "llm_invalid_output"),
    ],
)
def test_unusable_responses_are_classified(body, code):
    with pytest.raises(LLMError) as excinfo:
        make_client(body=body).generate_structured("s", "u", GeneratedAnswer)
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    ("status", "error_status", "message", "code", "retryable"),
    [
        (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "llm_auth_failed", False),
        (403, "PERMISSION_DENIED", "Permission denied", "llm_permission_denied", False),
        (404, "NOT_FOUND", "models/x is not found", "llm_model_not_found", False),
        (429, "RESOURCE_EXHAUSTED", "Quota exceeded for api key", "llm_rate_limited", True),
        (500, "INTERNAL", "Internal error", "llm_unavailable", True),
        (400, "INVALID_ARGUMENT", "Invalid JSON payload", "llm_bad_request", False),
    ],
)
def test_http_errors_are_classified(status, error_status, message, code, retryable):
    llm = make_client(status=status, error_status=error_status, error_message=message)
    with pytest.raises(LLMError) as excinfo:
        llm.generate_structured("s", "u", GeneratedAnswer)
    assert excinfo.value.code == code
    assert excinfo.value.retryable is retryable


def test_connection_failure_is_reported_as_unavailable():
    def handler(request):
        raise httpx.ConnectError("no network")

    llm = GeminiLLMClient("test-key", MODEL, "medium", 16000, 30.0)
    llm.client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
        client_args={"transport": httpx.MockTransport(handler)},
        retry_options=types.HttpRetryOptions(attempts=1),
    ))
    with pytest.raises(LLMError) as excinfo:
        llm.generate_structured("s", "u", GeneratedAnswer)
    assert excinfo.value.code == "llm_unavailable"
