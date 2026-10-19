"""Project-relative configuration; importing this module performs no network I/O."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """API keys are deliberately excluded from the object's representation."""

    root: Path = PROJECT_ROOT
    deepseek_api_key: str = field(default="", repr=False)
    qwen_api_key: str = field(default="", repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v3"
    timeout: float = 60.0
    top_k: int = 10
    recursion_limit: int = 40
    tracing: bool = False

    @property
    def data_path(self) -> Path:
        return self.root / "data" / "cat_codes_descriptions.csv"

    @property
    def runtime_path(self) -> Path:
        return self.root / "runtime"

    @property
    def vector_path(self) -> Path:
        return self.runtime_path / "chroma"

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "Settings":
        """Read an explicit .env, or the project's .env, without overriding the OS."""
        path = Path(env_file).expanduser().resolve() if env_file else PROJECT_ROOT / ".env"
        if env_file and not path.is_file():
            raise ValueError(f"找不到配置文件：{path}")
        load_dotenv(path, override=False)
        tracing = os.getenv("ENABLE_TRACING", "false").lower() in {"true", "1", "yes"}
        tracing_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if tracing and not tracing_key:
            raise ValueError("开启追踪前，请配置 LANGSMITH_API_KEY 或 LANGCHAIN_API_KEY。")
        os.environ["LANGSMITH_TRACING"] = str(tracing).lower()
        os.environ["LANGCHAIN_TRACING_V2"] = str(tracing).lower()
        if tracing:
            os.environ.setdefault("LANGSMITH_PROJECT", "Contextual-Quoting")
        settings = cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            qwen_api_key=os.getenv("QWEN_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", cls.deepseek_base_url),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", cls.deepseek_model),
            qwen_base_url=os.getenv("QWEN_BASE_URL", cls.qwen_base_url),
            qwen_model=os.getenv("QWEN_MODEL", cls.qwen_model),
            embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", cls.embedding_model),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "60")),
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
            recursion_limit=int(os.getenv("GRAPH_RECURSION_LIMIT", "40")),
            tracing=tracing,
        )
        if settings.timeout <= 0 or settings.top_k <= 0 or settings.recursion_limit < 5:
            raise ValueError("超时和检索数量必须为正数，GRAPH_RECURSION_LIMIT 至少为 5。")
        return settings

    def require_chat(self) -> None:
        """Validate only the credentials required for a chat model."""
        if not (self.deepseek_api_key or self.qwen_api_key):
            raise ValueError("请在项目 .env 中配置 DEEPSEEK_API_KEY 或 QWEN_API_KEY。")

    def require_embeddings(self) -> None:
        """Embeddings use Qwen even when DeepSeek handles conversation."""
        if not self.qwen_api_key:
            raise ValueError("分类检索需要 QWEN_API_KEY（千问嵌入模型）。")
