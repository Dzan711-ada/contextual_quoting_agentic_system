"""Parse the tutorial's line-based catalog without splitting classification rows."""

from dataclasses import dataclass
from decimal import Decimal
import re
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

ROW_PATTERN = re.compile(r"^CATEGORY:\s*(.*?), CODE:\s*(\d*), DESCRIPTION:\s*(.*), RATE:\s*([\d.]+)\s*$")


def normalized(value: str) -> str:
    """Ignore punctuation when checking a model's copy of a catalog description."""
    return " ".join(re.findall(r"\w+", value.casefold()))


@dataclass(frozen=True)
class Classification:
    category: str
    code: int
    description: str
    rate: Decimal

    @property
    def text(self) -> str:
        return f"CATEGORY: {self.category}, CODE: {self.code}, DESCRIPTION: {self.description}, RATE: {self.rate}"


class Catalog:
    """Codes are not unique; category, code and description form the identity."""

    def __init__(self, records: list[Classification], skipped_lines: list[int] | None = None):
        if not records:
            raise ValueError("分类目录为空。")
        self.records = records
        self.skipped_lines = skipped_lines or []
        self._lookup = {
            (r.category, r.code, normalized(r.description)): r for r in records
        }
        self.rates: dict[str, Decimal] = {}
        for record in records:
            if record.category in self.rates and self.rates[record.category] != record.rate:
                raise ValueError(f"同一类别存在冲突费率：{record.category}")
            self.rates[record.category] = record.rate

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        """The .csv is actually one labeled text record per line, without a header."""
        records, skipped = [], []
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            match = ROW_PATTERN.fullmatch(line.strip())
            if not match:
                raise ValueError(f"分类数据第 {number} 行格式无效。")
            category, code, description, rate = match.groups()
            if not code:
                skipped.append(number)
                continue
            amount = Decimal(rate)
            if not amount.is_finite() or amount <= 0:
                raise ValueError(f"分类数据第 {number} 行费率无效。")
            records.append(Classification(category.strip().upper(), int(code), description.strip(), amount))
        if skipped:
            logger.info("分类目录跳过缺少代码的数据行：%s", skipped)
        return cls(records, skipped)

    def match(self, category: str, code: int, description: str) -> Classification:
        """Reject invented categories and ambiguous code-only matches."""
        key = (category.strip().upper(), code, normalized(description))
        try:
            return self._lookup[key]
        except KeyError as exc:
            raise ValueError(f"分类不在目录中：{category} / {code} / {description}") from exc
