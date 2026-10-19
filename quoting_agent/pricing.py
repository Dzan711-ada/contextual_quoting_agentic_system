"""Deterministic sample pricing, using the catalog as the only rate source."""

from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping

from .models import RateRequest


CENT = Decimal("0.01")


def money(value: Decimal) -> str:
    return str(value.quantize(CENT, rounding=ROUND_HALF_UP))


class QuoteCalculator:
    """Use a private snapshot of catalog rates instead of a derived rate database."""

    def __init__(self, rates: Mapping[str, Decimal]):
        self._rates = {name.strip().upper(): Decimal(str(rate)) for name, rate in rates.items()}
        if not self._rates or not all(self._rates):
            raise ValueError("费率目录和类别名称不能为空。")
        if any(not rate.is_finite() or rate <= 0 for rate in self._rates.values()):
            raise ValueError("费率必须为有限正数。")

    def calculate(self, categories: list[str], projected_revenue: float) -> dict:
        request = RateRequest(categories=categories, projected_revenue=projected_revenue)
        names = list(dict.fromkeys(name.strip().upper() for name in request.categories))
        if not all(names):
            raise ValueError("类别名称不能为空。")
        revenue = Decimal(str(request.projected_revenue)).quantize(CENT, rounding=ROUND_HALF_UP)
        if revenue <= 0:
            raise ValueError("年营收至少为 0.01 美元。")
        missing = [name for name in names if name not in self._rates]
        if missing:
            raise ValueError(f"以下类别缺少费率，无法生成完整报价：{', '.join(missing)}")
        cents, remainder = divmod(int(revenue * 100), len(names))
        items, total = [], Decimal(0)
        for index, name in enumerate(names):
            portion = Decimal(cents + (index < remainder)) / 100
            rate = self._rates[name]
            premium = (portion / 1000 * rate).quantize(CENT, rounding=ROUND_HALF_UP)
            total += premium
            items.append({"category": name, "rate": str(rate), "allocated_revenue": money(portion), "premium": money(premium)})
        return {
            "currency": "USD", "projected_revenue": money(revenue),
            "allocation_method": "equal_by_unique_category", "items": items,
            "total_annual_premium": money(total),
        }


def format_quote(quote: dict) -> str:
    """Render authoritative numbers without asking a model to rewrite them."""
    lines = ["教学示例报价（USD）", f"预计年营收：${Decimal(quote['projected_revenue']):,.2f}"]
    for item in quote["items"]:
        lines.append(
            f"- {item['category']}: 每 $1,000 费率 ${item['rate']}；"
            f"分配营收 ${Decimal(item['allocated_revenue']):,.2f}；"
            f"年保费 ${Decimal(item['premium']):,.2f}"
        )
    lines.extend([
        f"总年保费：${Decimal(quote['total_annual_premium']):,.2f}",
        "规则：按去重后的类别平均分配营收，逐项四舍五入到美分。费率来自教程样例。",
    ])
    return "\n".join(lines)
