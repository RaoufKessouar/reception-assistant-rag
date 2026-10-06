from types import SimpleNamespace

from scripts.judge_generation_report import _judge


class _FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        content = (
            '{"verdict":"pass","score":4,"grounded":true,'
            '"relevant":true,"citations_correct":true,'
            '"abstention_appropriate":true,"reason":"preuves suffisantes"}'
        )
        usage = SimpleNamespace(
            model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_deepseek_judge_uses_thinking_and_json_mode():
    client = _FakeClient()

    decision, usage = _judge(
        client, "deepseek-v4-pro", {"question": "Test"}, provider="deepseek"
    )

    request = client.chat.completions.request
    assert decision.verdict == "pass"
    assert usage["prompt_tokens"] == 10
    assert request["response_format"] == {"type": "json_object"}
    assert request["reasoning_effort"] == "high"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in request


def test_local_judge_keeps_vllm_schema_constraint():
    client = _FakeClient()

    _judge(client, "local-model", {"question": "Test"})

    request = client.chat.completions.request
    assert "guided_json" in request["extra_body"]
    assert request["temperature"] == 0.0
