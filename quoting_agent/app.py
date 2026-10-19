"""Application assembly and per-conversation checkpoint management."""

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from .catalog import Catalog
from .config import Settings
from .llm import ModelFactory
from .pricing import QuoteCalculator
from .retrieval import CategoryRetriever
from .workflow import build_graph, reply


def create_graph(settings: Settings, *, catalog: Catalog | None = None):
    """One assembly path for CLI and live checks; reuse an already loaded catalog."""
    catalog = catalog if catalog is not None else Catalog.load(settings.data_path)
    return build_graph(
        ModelFactory(settings), CategoryRetriever(settings, catalog), catalog,
        QuoteCalculator(catalog.rates),
    )


class Conversation:
    """History stays in memory for this process; distinct thread IDs are isolated."""

    def __init__(self, graph, recursion_limit: int = 40):
        self.graph = graph
        self.recursion_limit = recursion_limit
        self.reset()

    def reset(self) -> None:
        """Start a clean conversation without deleting data or the vector cache."""
        self.config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": self.recursion_limit,
        }

    def ask(self, text: str) -> list[str]:
        """Return the final user-facing reply while keeping the full graph history."""
        if not text.strip():
            return []
        before = self.graph.get_state(self.config).values.get("messages", [])
        old_ids = {message.id for message in before}
        try:
            state = self.graph.invoke({"messages": [HumanMessage(content=text)]}, config=self.config)
        except Exception:
            # A failed API call can leave a handoff tool unanswered in a checkpoint.
            # Close every pending call before accepting another human message.
            snapshot = self.graph.get_state(self.config)
            messages = snapshot.values.get("messages", [])
            answered = {m.tool_call_id for m in messages if m.type == "tool"}
            pending = [
                call for message in messages for call in getattr(message, "tool_calls", [])
                if call["id"] not in answered
            ]
            recovery = [reply(call, "The operation failed. Ask the user to retry; no quote was completed.") for call in pending]
            recovery.append(AIMessage(content="本轮处理未完成，请检查服务配置后重试，或输入 /new 开始新会话。"))
            self.graph.update_state(
                self.config, {"messages": recovery, "workflow_state": "main_assistant"},
                as_node="main_assistant",
            )
            raise
        visible = [
            message for message in state["messages"]
            if isinstance(message, AIMessage) and message.id not in old_ids and message.text().strip()
        ]
        # Keep internal tool preambles in history, but show one final answer per turn.
        return [visible[-1].text()] if visible else []

    @property
    def state(self) -> dict:
        return self.graph.get_state(self.config).values
