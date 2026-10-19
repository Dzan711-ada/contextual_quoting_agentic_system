"""Enter-to-send, paste batching and process shutdown checks without paid APIs."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from queue import Queue
import subprocess
import sys
import unittest
from unittest.mock import patch

from quoting_agent.cli import ConsoleInput, main, run_chat
from quoting_agent.config import Settings


ROOT = Path(__file__).resolve().parents[1]
PARAGRAPHS = "Hosted AI applications.\n\nAI strategy and IT consulting.\n\nAnnual revenue is USD 4,000,000.\n"


class FeedStream:
    """Keep stdin open while a test enters a line, as in an IDE console."""

    def __init__(self):
        self.events = Queue()

    def readline(self):
        event = self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event


class FakeConversation:
    def __init__(self, *args):
        self.requests = []
        self.resets = 0

    def ask(self, text):
        self.requests.append(text)
        return ["收到完整消息。"]

    def reset(self):
        self.resets += 1


class InputTests(unittest.TestCase):
    def test_enter_submits_while_stdin_remains_open(self):
        stream = FeedStream()
        reader = ConsoleInput(stream, io.StringIO(), paste_gap=0.005)
        try:
            stream.events.put("你好\n")
            self.assertEqual(reader.read_message(), "你好")
        finally:
            reader.close()
            stream.events.put("")

    def test_manually_entered_turns_remain_separate(self):
        stream = FeedStream()
        reader = ConsoleInput(stream, io.StringIO(), paste_gap=0.005)
        try:
            stream.events.put("First\n")
            self.assertEqual(reader.read_message(), "First")
            stream.events.put("Second\n")
            self.assertEqual(reader.read_message(), "Second")
        finally:
            reader.close()
            stream.events.put("")

    def test_pasted_paragraphs_are_one_message_without_submit_command(self):
        reader = ConsoleInput(io.StringIO(PARAGRAPHS), io.StringIO())
        self.assertEqual(reader.read_message(), PARAGRAPHS.strip())
        reader.close()

    def test_empty_lines_do_not_repeat_input_prompt(self):
        output = io.StringIO()
        reader = ConsoleInput(io.StringIO("\n \n\n你好\n"), output)
        self.assertEqual(reader.read_message(), "你好")
        self.assertEqual(output.getvalue().count("你："), 1)
        reader.close()

    def test_commands_form_boundaries_between_messages(self):
        reader = ConsoleInput(io.StringIO("First\n/new\nSecond\n/exit\n"), io.StringIO())
        self.assertEqual([reader.read_message() for _ in range(4)], ["First", "/new", "Second", "/exit"])
        reader.close()

    def test_case_insensitive_commands(self):
        reader = ConsoleInput(io.StringIO(" /EXIT \n"), io.StringIO())
        self.assertEqual(reader.read_message(), "/exit")
        reader.close()

    def test_blank_stdin_ends_without_any_message(self):
        output = io.StringIO()
        reader = ConsoleInput(io.StringIO("\n \n"), output)
        with self.assertRaises(EOFError):
            reader.read_message()
        with self.assertRaises(EOFError):
            reader.read_message()
        self.assertEqual(output.getvalue().count("你："), 1)
        reader.close()

    def test_input_interrupt_propagates(self):
        stream = FeedStream()
        reader = ConsoleInput(stream, io.StringIO())
        stream.events.put(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            reader.read_message()
        reader.close()


class ChatTests(unittest.TestCase):
    def run_session(self, text, conversation=None):
        conversation = conversation or FakeConversation()
        output = io.StringIO()
        reader = ConsoleInput(io.StringIO(text), output)
        with redirect_stdout(output), redirect_stderr(output):
            run_chat(conversation, reader=reader)
        return conversation, output.getvalue()

    def test_three_pasted_paragraphs_make_one_model_call(self):
        conversation, output = self.run_session(PARAGRAPHS + "/exit\n")
        self.assertEqual(conversation.requests, [PARAGRAPHS.strip()])
        self.assertEqual(output.count("你："), 2)
        self.assertEqual(output.count("助手："), 1)

    def test_empty_input_makes_no_model_call(self):
        conversation, output = self.run_session("\n\n/exit\n")
        self.assertEqual(conversation.requests, [])
        self.assertEqual(output.count("你："), 1)

    def test_new_is_handled_locally(self):
        conversation, _ = self.run_session("/new\n/exit\n")
        self.assertEqual(conversation.resets, 1)
        self.assertEqual(conversation.requests, [])

    def test_service_failure_preserves_next_turn_and_hides_error_body(self):
        conversation = FakeConversation()
        with patch.object(conversation, "ask", side_effect=[ConnectionError("private body"), ["Recovered"]]) as ask:
            _, output = self.run_session("First\n/new\nSecond\n/exit\n", conversation)
        self.assertEqual(ask.call_count, 2)
        self.assertIn("Recovered", output)
        self.assertNotIn("private body", output)

    def test_cli_uses_enter_mode_without_extra_arguments(self):
        output = io.StringIO()
        reader = ConsoleInput(io.StringIO(PARAGRAPHS + "/exit\n"), output)
        conversation = FakeConversation()
        settings = Settings(root=ROOT, qwen_api_key="unused-test-key")
        with patch("quoting_agent.cli.Settings.load", return_value=settings), patch("quoting_agent.app.create_graph", return_value=object()), patch("quoting_agent.app.Conversation", return_value=conversation), patch("quoting_agent.cli.ConsoleInput", return_value=reader), redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertEqual(conversation.requests, [PARAGRAPHS.strip()])
        self.assertIn("回车发送", output.getvalue())
        self.assertNotIn("/send", output.getvalue())


class ProcessTests(unittest.TestCase):
    def test_process_submits_on_enter_and_exits_with_parent_stdin_open(self):
        script = (
            "import sys; sys.stdout.reconfigure(encoding='utf-8'); "
            "from quoting_agent.cli import ConsoleInput; "
            "reader=ConsoleInput(); print(reader.read_message(), flush=True); reader.close()"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        try:
            process.stdin.write("Hello\n")
            process.stdin.flush()
            # Do not close stdin: an extra command or EOF must not be required.
            process.wait(timeout=10)
            output = process.stdout.read()
            error = process.stderr.read()
            self.assertEqual(process.returncode, 0, error)
            self.assertIn("Hello", output)
            self.assertNotIn("Fatal Python error", error)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def test_offline_demo_does_not_import_graph_or_create_rate_database(self):
        script = (
            "import sys; from quoting_agent.cli import main; "
            "assert main(['demo'])==0; assert 'langgraph' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("7,880.00", result.stdout)
        self.assertNotIn("LangChainPendingDeprecationWarning", result.stderr)


if __name__ == "__main__":
    unittest.main()
