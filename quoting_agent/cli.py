"""Command line entry point, Enter-to-send input, and interactive output."""

import argparse
from collections import deque
import logging
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Thread
from typing import TextIO

from .catalog import Catalog
from .config import Settings
from .pricing import QuoteCalculator, format_quote


_END = object()
_COMMANDS = {"/new", "/exit"}


class ConsoleInput:
    """Send on Enter and coalesce lines that arrive together from a paste.

    A short idle gap groups already arriving lines, without a submit command.
    Input is read in one place so Python's own buffering cannot hide pasted lines
    from an OS-level readiness check. The worker never calls a model or prints.
    """

    def __init__(self, stream: TextIO | None = None, output: TextIO | None = None, *, paste_gap: float = 0.08):
        self.stream = stream if stream is not None else sys.stdin
        self.output = output if output is not None else sys.stdout
        self.paste_gap = paste_gap
        self._queue = Queue()
        self._pending = deque()
        self._stop = Event()
        self._worker = None
        self._ended = False

    def _read_lines(self) -> None:
        try:
            while not self._stop.is_set():
                line = self.stream.readline()
                if line == "":
                    break
                self._queue.put(line.rstrip("\r\n"))
        except BaseException as error:
            self._queue.put(error)
        finally:
            self._queue.put(_END)

    def _next(self, timeout: float | None = None):
        if self._pending:
            return self._pending.popleft()
        if timeout is not None:
            return self._queue.get(timeout=timeout)
        # A finite queue wait lets Ctrl+C interrupt the main thread on Windows.
        while True:
            try:
                return self._queue.get(timeout=0.1)
            except Empty:
                pass

    def read_message(self) -> str:
        if self._ended:
            raise EOFError
        print("\n你：", end="", file=self.output, flush=True)
        if self._worker is None:
            self._worker = Thread(target=self._read_lines, name="console-input", daemon=True)
            self._worker.start()
        parts = []
        while True:
            line = self._next()
            if line is _END:
                self._ended = True
                raise EOFError
            if isinstance(line, BaseException):
                raise line
            if line.strip():
                break
        command = line.strip().lower()
        if command in _COMMANDS:
            return command
        parts.append(line)
        while True:
            try:
                line = self._next(timeout=self.paste_gap)
            except Empty:
                break
            if isinstance(line, BaseException):
                raise line
            if line is _END or line.strip().lower() in _COMMANDS:
                self._pending.appendleft(line)
                break
            parts.append(line)
        return "\n".join(parts).strip()

    def close(self) -> None:
        """Leave the IDE's standard input open; a blocked reader exits with the app."""
        self._stop.set()


def run_chat(conversation, *, reader: ConsoleInput | None = None) -> None:
    reader = reader if reader is not None else ConsoleInput()
    print("输入后直接按回车发送。/new 开始新会话，/exit 退出。")
    try:
        while True:
            try:
                text = reader.read_message()
            except (EOFError, KeyboardInterrupt):
                print("\n会话已结束。")
                return
            if text == "/exit":
                return
            if text == "/new":
                conversation.reset()
                print("已开始新会话。")
                continue
            try:
                for response in conversation.ask(text):
                    print(f"\n助手：{response}")
            except Exception as error:
                print(f"\n本轮处理失败（{type(error).__name__}），请检查服务配置后重试；/new 可重置会话。", file=sys.stderr)
    finally:
        reader.close()


def main(argv: list[str] | None = None) -> int:
    """Run main.py in PyCharm, or select an offline/maintenance command."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="上下文报价多智能体教学项目")
    parser.add_argument("command", nargs="?", choices=["chat", "check", "index", "demo"], default="chat")
    parser.add_argument("--env-file", type=Path, help="指定 .env 路径；默认使用本项目 .env")
    parser.add_argument("--verbose", action="store_true", help="显示索引进度和数据诊断")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = Settings.load(args.env_file)
        catalog = Catalog.load(settings.data_path)
        if args.command == "demo":
            print(format_quote(QuoteCalculator(catalog.rates).calculate(["CONSULTANTS"], 4_000_000)))
            return 0
        if args.command == "check":
            print(f"项目目录：{settings.root}")
            print(f"有效分类：{len(catalog.records)}；跳过缺码记录：{len(catalog.skipped_lines)}；费率类别：{len(catalog.rates)}")
            print(f"DeepSeek 对话：{'已配置' if settings.deepseek_api_key else '未配置'}")
            print(f"千问对话及嵌入：{'已配置' if settings.qwen_api_key else '未配置'}")
            print(f"追踪：{'开启' if settings.tracing else '关闭'}")
            settings.require_chat()
            settings.require_embeddings()
            print("配置检查通过（未调用远程模型）。")
            return 0
        if args.command == "index":
            from .retrieval import CategoryRetriever
            count = CategoryRetriever(settings, catalog).prepare()
            print(f"已准备 {count} 条分类向量；缓存位置：{settings.vector_path}")
            return 0
        settings.require_chat()
        settings.require_embeddings()
        from .app import Conversation, create_graph
        conversation = Conversation(create_graph(settings, catalog=catalog), settings.recursion_limit)
        print("上下文报价助手已启动。")
        print(f"分类目录包含 {len(catalog.records)} 条记录，检索时优先复用本地向量缓存。")
        print("请介绍企业名称、业务、行业、经营方式和预计年营收（USD）。")
        run_chat(conversation)
        return 0
    except (OSError, ValueError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"启动失败（{type(error).__name__}），请检查依赖和服务配置。", file=sys.stderr)
        return 1
