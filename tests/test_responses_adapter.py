# encoding: utf-8

"""Unit tests for the OpenAI Responses <-> Chat Completions adapter.

Run:
    python -m unittest tests.test_responses_adapter -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.openai.responses_adapter import (
    build_responses_request,
    translate_stream,
    translate_sync,
)


class TestBuildResponsesRequest(unittest.TestCase):
    def test_basic_user_message(self):
        chat = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 100,
        }
        out = build_responses_request(chat)
        self.assertEqual(out["model"], "gpt-5.5")
        self.assertEqual(out["stream"], True)
        self.assertNotIn("temperature", out)
        self.assertEqual(out["max_output_tokens"], 100)
        self.assertNotIn("max_tokens", out)
        self.assertEqual(len(out["input"]), 1)
        self.assertEqual(out["input"][0]["role"], "user")
        self.assertEqual(out["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(out["input"][0]["content"][0]["text"], "hi")

    def test_system_message_becomes_instructions(self):
        chat = {
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
            ],
        }
        out = build_responses_request(chat)
        self.assertEqual(out["instructions"], "you are helpful")
        # instructions consumed -> input has only the user msg
        self.assertEqual(len(out["input"]), 1)
        self.assertEqual(out["input"][0]["role"], "user")

    def test_tools_flattened(self):
        chat = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo back",
                    "parameters": {"type": "object", "properties": {"t": {"type": "string"}}},
                },
            }],
            "tool_choice": "auto",
        }
        out = build_responses_request(chat)
        self.assertEqual(len(out["tools"]), 1)
        t = out["tools"][0]
        self.assertEqual(t["type"], "function")
        self.assertEqual(t["name"], "echo")
        self.assertEqual(t["description"], "echo back")
        self.assertIn("parameters", t)
        # Nested {"function": {...}} flattened away
        self.assertNotIn("function", t)
        self.assertEqual(out["tool_choice"], "auto")

    def test_assistant_with_tool_calls_to_function_call_items(self):
        chat = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "calling tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"t":"x"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "x"},
            ],
        }
        out = build_responses_request(chat)
        items = out["input"]
        # user message + assistant text message + function_call item + function_call_output
        self.assertEqual(len(items), 4)
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[1]["type"], "message")
        self.assertEqual(items[1]["role"], "assistant")
        self.assertEqual(items[1]["content"][0]["type"], "output_text")
        self.assertEqual(items[2]["type"], "function_call")
        self.assertEqual(items[2]["call_id"], "call_1")
        self.assertEqual(items[2]["name"], "echo")
        self.assertEqual(items[2]["arguments"], '{"t":"x"}')
        self.assertEqual(items[3]["type"], "function_call_output")
        self.assertEqual(items[3]["call_id"], "call_1")
        self.assertEqual(items[3]["output"], "x")

    def test_reasoning_effort_passthrough(self):
        out = build_responses_request(
            {"messages": [{"role": "user", "content": "hi"}]},
            reasoning_effort="high",
        )
        self.assertEqual(out["reasoning"], {"effort": "high"})

    def test_no_reasoning_when_unset(self):
        out = build_responses_request({"messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("reasoning", out)

    def test_list_content_blocks_flattened(self):
        chat = {"messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }]}
        out = build_responses_request(chat)
        self.assertEqual(out["input"][0]["content"][0]["text"], "ab")


class TestTranslateStream(unittest.TestCase):
    def test_text_delta(self):
        chunks = [
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.output_text.delta", "delta": "!"},
            {"type": "response.completed", "response": {"output": [], "usage": {}}},
        ]
        out = list(translate_stream(chunks))
        # 2 text chunks + 1 finish chunk
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["choices"][0]["delta"]["content"], "Hi")
        self.assertEqual(out[1]["choices"][0]["delta"]["content"], "!")
        self.assertEqual(out[2]["choices"][0]["finish_reason"], "stop")

    def test_function_call_assembly(self):
        chunks = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_x", "name": "echo", "arguments": ""},
            },
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"t"'},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ':"hi"}'},
            {
                "type": "response.completed",
                "response": {
                    "output": [{"type": "function_call", "call_id": "call_x", "name": "echo",
                                "arguments": '{"t":"hi"}'}],
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            },
        ]
        out = list(translate_stream(chunks))
        # output_item.added emits a tool_call seed (id+name); 2 deltas + completed
        self.assertEqual(len(out), 4)
        seed = out[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(seed["index"], 0)
        self.assertEqual(seed["id"], "call_x")
        self.assertEqual(seed["function"]["name"], "echo")
        self.assertEqual(out[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], '{"t"')
        self.assertEqual(out[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], ':"hi"}')
        finish = out[3]
        self.assertEqual(finish["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(finish["usage"]["prompt_tokens"], 5)
        self.assertEqual(finish["usage"]["completion_tokens"], 10)
        self.assertEqual(finish["usage"]["total_tokens"], 15)

    def test_error_chunk_passthrough(self):
        chunks = [{"error": True, "message": "boom", "status_code": 500}]
        out = list(translate_stream(chunks))
        self.assertEqual(out, chunks)

    def test_response_failed(self):
        chunks = [{
            "type": "response.failed",
            "response": {"error": {"message": "nope", "code": "x", "type": "y"}},
        }]
        out = list(translate_stream(chunks))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["error"]["message"], "nope")
        self.assertEqual(out[0]["error"]["code"], "x")

    def test_unknown_events_dropped(self):
        chunks = [
            {"type": "response.created"},
            {"type": "response.in_progress"},
            {"type": "response.output_text.delta", "delta": "x"},
            {"type": "response.content_part.added"},
        ]
        out = list(translate_stream(chunks))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["choices"][0]["delta"]["content"], "x")


class TestTranslateSync(unittest.TestCase):
    def test_text_only(self):
        resp = {
            "id": "resp_1",
            "model": "gpt-5.5",
            "created_at": 1234,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Hi!"}],
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        out = translate_sync(resp)
        self.assertEqual(out["choices"][0]["message"]["content"], "Hi!")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["total_tokens"], 5)

    def test_function_call(self):
        resp = {
            "output": [{
                "type": "function_call",
                "call_id": "call_x",
                "name": "echo",
                "arguments": '{"t":"hi"}',
            }],
            "usage": {},
        }
        out = translate_sync(resp)
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["content"], "")
        self.assertEqual(len(msg["tool_calls"]), 1)
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["id"], "call_x")
        self.assertEqual(tc["function"]["name"], "echo")
        self.assertEqual(tc["function"]["arguments"], '{"t":"hi"}')
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
