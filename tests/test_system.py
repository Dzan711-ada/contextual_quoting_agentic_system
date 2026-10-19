"""Behavioral checks use scripted models: no API credentials or network required."""

from collections import deque
from decimal import Decimal
from pathlib import Path
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from quoting_agent.app import Conversation
from quoting_agent.catalog import Catalog, Classification
from quoting_agent.config import PROJECT_ROOT, Settings
from quoting_agent.llm import ModelFactory
from quoting_agent.models import CategoriesOutput, DescriptionGrade
from quoting_agent.pricing import QuoteCalculator
from quoting_agent.workflow import Assistant, build_graph


INFO = {
    "legal_name": "GABAS, LLC", "primary_products_or_services": "Hosted custom AI applications",
    "secondary_or_ancillary_operations": "AI strategy consulting", "industries": "Technology",
    "manufacturing_retail_wholesale_or_distribution": "Software services, mostly remote",
    "projected_revenue": 4_000_000,
}


def call(name, args, key, content=""):
    return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": key}])


def sample_catalog():
    return Catalog([
        Classification("CONSULTANTS", 68673, "Computer Consultants--Application Service Provider.", Decimal("1.97")),
        Classification("TECHNOLOGY", 10001, "Software Services.", Decimal("8.57")),
    ])


class ScriptedFactory:
    """Use role-specific response queues to test the actual compiled graph."""

    def __init__(self, catalog):
        self.queues = {name: deque() for name in ("main", "underwriting", "quote")}
        self.record = catalog.records[0]

    def tools(self, tools):
        names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools}
        role = "main" if "ToUnderwritingAssistant" in names else "underwriting" if "ToQuoteAssistant" in names else "quote"

        def respond(payload):
            if not self.queues[role]:
                raise AssertionError(f"Unexpected extra {role} call")
            return self.queues[role].popleft()

        return RunnableLambda(respond)

    def chat(self):
        return RunnableLambda(lambda _: AIMessage(content="Hosted AI software and consulting services."))

    def structured(self, schema):
        r = self.record
        if schema is CategoriesOutput:
            result = {"categories": [{"category": r.category, "cat_description": r.description, "code": r.code, "rate": 999, "rationalization": "Hosted applications."}]}
        else:
            result = {"description_grades": [{"category": r.category, "cat_description": r.description, "code": r.code, "grade": 0.95}]}
        return RunnableLambda(lambda _: schema.model_validate(result))


class StaticRetriever:
    def __init__(self, catalog):
        self.catalog = catalog

    def invoke(self, text, config=None):
        return [record.text for record in self.catalog.records]


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.store = QuoteCalculator(sample_catalog().rates)

    def test_original_example_and_duplicate_categories(self):
        result = self.store.calculate(["CONSULTANTS", " consultants "], 4_000_000)
        self.assertEqual(result["total_annual_premium"], "7880.00")
        self.assertEqual(len(result["items"]), 1)

    def test_rate_snapshot_does_not_follow_external_mutation(self):
        rates = {"CONSULTANTS": Decimal("1.97")}
        calculator = QuoteCalculator(rates)
        rates["CONSULTANTS"] = Decimal("999")
        self.assertEqual(calculator.calculate(["CONSULTANTS"], 4_000_000)["total_annual_premium"], "7880.00")

    def test_unknown_category_cannot_produce_partial_quote(self):
        with self.assertRaises(ValueError):
            self.store.calculate(["CONSULTANTS", "UNKNOWN"], 1_000_000)

    def test_invalid_revenue_and_empty_categories(self):
        for revenue in (0, -100, float("inf"), float("nan")):
            with self.subTest(revenue=revenue), self.assertRaises(ValueError):
                self.store.calculate(["CONSULTANTS"], revenue)
        with self.assertRaises(ValueError):
            self.store.calculate([], 1000)

    def test_allocations_and_premiums_sum_exactly(self):
        result = self.store.calculate(["CONSULTANTS", "TECHNOLOGY"], 1000.01)
        self.assertEqual(sum(Decimal(i["allocated_revenue"]) for i in result["items"]), Decimal("1000.01"))
        self.assertEqual(sum(Decimal(i["premium"]) for i in result["items"]), Decimal(result["total_annual_premium"]))

    def test_bakery_example_uses_one_category_despite_multiple_codes(self):
        catalog = Catalog.load(PROJECT_ROOT / "data" / "cat_codes_descriptions.csv")
        result = QuoteCalculator(catalog.rates).calculate(["RESTAURANTS/MOBILE FOOD SERVICES"] * 2, 500_000)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["total_annual_premium"], "1425.00")

    def test_invalid_rate_catalog_is_rejected(self):
        for rates in ({}, {"": Decimal("1")}, {"A": Decimal("NaN")}, {"A": Decimal("-1")}):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                QuoteCalculator(rates)


class CatalogTests(unittest.TestCase):
    def test_original_data_skips_missing_code_and_preserves_long_rows(self):
        catalog = Catalog.load(PROJECT_ROOT / "data" / "cat_codes_descriptions.csv")
        self.assertEqual(len(catalog.records), 438)
        self.assertEqual(catalog.skipped_lines, [182])
        self.assertEqual(len(catalog.rates), 13)
        self.assertTrue(any(len(r.text) > 150 for r in catalog.records))
        self.assertGreater(len([r for r in catalog.records if r.code == 47367]), 1)

    def test_identity_includes_description(self):
        catalog = sample_catalog()
        self.assertEqual(catalog.match("consultants", 68673, "Computer Consultants – Application Service Provider").code, 68673)
        with self.assertRaises(ValueError):
            catalog.match("CONSULTANTS", 68673, "An invented description")

    def test_grade_range(self):
        with self.assertRaises(ValidationError):
            DescriptionGrade(category="CONSULTANTS", cat_description="x", code=1, grade=1.01)


class FallbackTests(unittest.TestCase):
    def test_native_fallback_preserves_config(self):
        def broken(value):
            raise ConnectionError("simulated")

        fallback = RunnableLambda(lambda value, config: (value, config["configurable"]["example"]))
        runnable = ModelFactory._combine([RunnableLambda(broken), fallback])
        self.assertEqual(runnable.with_config(configurable={"example": 7}).invoke("input"), ("input", 7))

    def test_stream_failure_before_first_chunk_uses_fallback(self):
        def broken(value):
            raise ConnectionError("simulated")
            yield value

        runnable = ModelFactory._combine([RunnableLambda(broken), RunnableLambda(lambda value: "backup")])
        self.assertEqual(list(runnable.stream("input")), ["backup"])

    def test_structured_output_explicitly_uses_function_calling(self):
        class Fake:
            def with_structured_output(self, schema, **kwargs):
                self.kwargs = kwargs
                return RunnableLambda(lambda _: schema)
        factory = object.__new__(ModelFactory)
        fake = Fake()
        factory.models = [fake]
        factory.structured(CategoriesOutput)
        self.assertEqual(fake.kwargs, {"method": "function_calling"})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.catalog = sample_catalog()
        self.rates = QuoteCalculator(self.catalog.rates)
        self.factory = ScriptedFactory(self.catalog)
        self.graph = build_graph(self.factory, StaticRetriever(self.catalog), self.catalog, self.rates)
        self.chat = Conversation(self.graph)

    def begin(self):
        self.factory.queues["main"].extend([
            call("info_tool", {"business_info": INFO}, "info-1"),
            call("ToUnderwritingAssistant", {"request": "Review the summarized business"}, "underwriting-1"),
        ])
        self.factory.queues["underwriting"].append(AIMessage(content="Please confirm hosted software and consulting."))
        self.chat.ask("Initial business description")

    def quote_handoff(self):
        r = self.catalog.records[0]
        return call("ToQuoteAssistant", {
            "request": "Confirmed", "final_classifications": [{"category": r.category, "code": r.code, "classification_description": r.description}],
        }, "quote-handoff")

    def assert_all_calls_answered(self):
        messages = self.chat.state["messages"]
        requested = [c["id"] for m in messages for c in getattr(m, "tool_calls", [])]
        answered = [m.tool_call_id for m in messages if m.type == "tool"]
        self.assertCountEqual(requested, answered)

    def test_two_turn_underwriting_quote_and_followup(self):
        self.begin()
        self.assertEqual(self.chat.state["workflow_state"], "underwriting_assistant")
        # Model-provided rates are overwritten by the canonical catalog rate.
        self.assertEqual(self.chat.state["rationale"][0]["rate"], 1.97)
        self.factory.queues["underwriting"].extend([
            call("info_tool", {"business_info": {"projected_revenue": 4_000_000}}, "info-2"),
            self.quote_handoff(),
        ])
        self.factory.queues["quote"].extend([
            call("fetch_rates", {"categories": ["CONSULTANTS"], "projected_revenue": 4_000_000}, "rates-1", "Let me calculate."),
            call("CompleteOrEscalate", {"cancel": True, "reason": "Quote complete"}, "complete-1", "Here is the model's quote summary."),
        ])
        responses = self.chat.ask("Confirmed; keep revenue at 4 million.")
        self.assertEqual(len(responses), 1)
        self.assertIn("7,880.00", responses[-1])
        self.assertEqual(self.chat.state["business_information"]["legal_name"], INFO["legal_name"])
        self.assertEqual(self.chat.state["workflow_state"], "main_assistant")
        self.assert_all_calls_answered()
        self.factory.queues["main"].append(AIMessage(content="The sample quote is in USD."))
        self.assertEqual(self.chat.ask("What currency?"), ["The sample quote is in USD."])

    def test_parallel_calls_get_matching_errors_without_execution(self):
        self.factory.queues["main"].extend([
            AIMessage(content="", tool_calls=[{"name": "info_tool", "args": {"business_info": INFO}, "id": "parallel-1"}, {"name": "ToUnderwritingAssistant", "args": {"request": "x"}, "id": "parallel-2"}]),
            AIMessage(content="Let me handle one operation at a time."),
        ])
        self.chat.ask("hello")
        self.assertFalse(self.chat.state.get("business_information"))
        self.assert_all_calls_answered()

    def test_missing_facts_block_handoff(self):
        self.factory.queues["main"].extend([
            call("ToUnderwritingAssistant", {"request": "Review"}, "missing"),
            AIMessage(content="What is the business name and annual revenue?"),
        ])
        self.chat.ask("hello")
        self.assertFalse(self.chat.state.get("documents"))
        self.assert_all_calls_answered()

    def test_changed_revenue_is_rejected_then_repaired(self):
        self.begin()
        self.factory.queues["underwriting"].append(self.quote_handoff())
        self.factory.queues["quote"].extend([
            call("fetch_rates", {"categories": ["TECHNOLOGY"], "projected_revenue": 4_000_000}, "wrong-category"),
            call("fetch_rates", {"categories": ["CONSULTANTS"], "projected_revenue": 5_000_000}, "wrong-revenue"),
            call("fetch_rates", {"categories": ["CONSULTANTS"], "projected_revenue": 4_000_000}, "right-revenue"),
            call("CompleteOrEscalate", {"cancel": True, "reason": "done"}, "complete"),
        ])
        self.chat.ask("Confirmed")
        self.assertEqual(self.chat.state["last_quote"]["total_annual_premium"], "7880.00")
        errors = [m.content for m in self.chat.state["messages"] if m.type == "tool" and "rejected" in m.content]
        self.assertEqual(len(errors), 2)
        self.assert_all_calls_answered()

    def test_repeated_tool_call_reuses_calculation(self):
        self.begin()
        self.factory.queues["underwriting"].append(self.quote_handoff())
        self.factory.queues["quote"].extend([
            call("fetch_rates", {"categories": ["CONSULTANTS"], "projected_revenue": 4_000_000}, "rate-a"),
            call("fetch_rates", {"categories": ["CONSULTANTS"], "projected_revenue": 4_000_000}, "rate-b"),
            call("CompleteOrEscalate", {"cancel": True, "reason": "done"}, "complete"),
        ])
        with patch.object(self.rates, "calculate", wraps=self.rates.calculate) as calculate:
            self.chat.ask("Confirmed")
        self.assertEqual(calculate.call_count, 1)
        self.assert_all_calls_answered()

    def test_failure_closes_pending_handoff_and_can_retry(self):
        self.factory.queues["main"].extend([
            call("info_tool", {"business_info": INFO}, "info"),
            call("ToUnderwritingAssistant", {"request": "Review"}, "pending"),
        ])
        with patch.object(StaticRetriever, "invoke", side_effect=ConnectionError("simulated")):
            with self.assertRaises(ConnectionError):
                self.chat.ask("hello")
        self.assert_all_calls_answered()
        self.factory.queues["main"].append(AIMessage(content="Ready to retry."))
        self.assertEqual(self.chat.ask("retry"), ["Ready to retry."])

    def test_reset_isolates_conversations(self):
        self.begin()
        self.chat.reset()
        self.assertFalse(self.chat.state)
        self.factory.queues["main"].append(AIMessage(content="Please introduce the new business."))
        self.chat.ask("new company")
        self.assertFalse(self.chat.state.get("business_information"))

    def test_empty_response_retry_is_bounded(self):
        assistant = Assistant(RunnableLambda(lambda _: AIMessage(content="")))
        with self.assertRaises(RuntimeError):
            assistant({"messages": [HumanMessage(content="hello")]}, {})


class ConfigurationTests(unittest.TestCase):
    def test_paths_do_not_depend_on_current_working_directory(self):
        settings = Settings(root=Path("X:/example"), deepseek_api_key="hidden", qwen_api_key="also-hidden")
        self.assertEqual(settings.data_path, Path("X:/example/data/cat_codes_descriptions.csv"))
        self.assertNotIn("hidden", repr(settings))

    def test_qwen_only_chat_and_embedding_validation(self):
        settings = Settings(qwen_api_key="unused-test-key")
        settings.require_chat()
        settings.require_embeddings()
        with self.assertRaises(ValueError):
            Settings(deepseek_api_key="unused-test-key").require_embeddings()


if __name__ == "__main__":
    unittest.main()
