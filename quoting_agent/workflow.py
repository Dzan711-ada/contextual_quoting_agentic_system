"""LangGraph orchestration of collection, retrieval, underwriting and quoting."""

from decimal import Decimal
import json
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .catalog import Catalog, normalized
from .models import (
    BusinessInformation, CategoriesOutput, CompleteOrEscalate,
    DescriptionGradeSchema, RateRequest,
    ToQuoteAssistant, ToUnderwritingAssistant,
)
from .pricing import QuoteCalculator, format_quote
from .prompts import (
    GRADING_PROMPT, MAIN_PROMPT, QUOTE_PROMPT, REASONING_PROMPT,
    REWRITE_PROMPT, UNDERWRITING_PROMPT,
)


class MainState(TypedDict, total=False):
    """Every intermediate value is declared so checkpoints retain the whole workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    business_information: dict[str, Any]
    workflow_state: Literal["main_assistant", "underwriting_assistant", "quote_assistant"]
    description: str
    documents: list[str]
    rationale: list[dict[str, Any]]
    graded_categories: list[dict[str, Any]]
    final_classifications: list[dict[str, Any]]
    tool_call_id: str
    last_quote: dict[str, Any]


@tool
def info_tool(business_info: BusinessInformation) -> dict:
    """Store only the newly supplied business facts; omitted fields are preserved."""
    return business_info.model_dump(exclude_none=True, exclude_unset=True)


@tool
def fetch_rates(categories: list[str], projected_revenue: float) -> dict:
    """Calculate sample annual premium for distinct confirmed categories in USD."""
    # Tool schemas are bound to the model; execution below has access to state.
    return RateRequest(categories=categories, projected_revenue=projected_revenue).model_dump()


class Assistant:
    """Bounded empty-response retries, with the graph's config forwarded intact."""

    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: MainState, config: RunnableConfig) -> dict:
        payload = {
            "messages": list(state.get("messages", [])),
            "business_information": state.get("business_information", {}),
            "graded_categories": state.get("graded_categories", []),
            "final_classifications": state.get("final_classifications", []),
            "last_quote": state.get("last_quote", {}),
        }
        for _ in range(3):
            result = self.runnable.invoke(payload, config=config)
            if not isinstance(result, AIMessage):
                raise TypeError("模型应返回 AIMessage。")
            if result.tool_calls or result.text().strip():
                return {"messages": [result]}
            payload["messages"].append(HumanMessage(content="Please provide a non-empty response or a tool call."))
        raise RuntimeError("模型连续三次返回空响应。")


def reply(call: dict, content: Any) -> ToolMessage:
    """Every tool invocation receives exactly one matching tool response."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ToolMessage(content=text, tool_call_id=call["id"], name=call["name"])


def missing_fields(info: dict) -> list[str]:
    """The six original collection fields must have explicit user-provided values."""
    return [field for field in BusinessInformation.model_fields if not info.get(field)]


def build_graph(factory, retriever, catalog: Catalog, rates: QuoteCalculator, checkpointer=None):
    """Construct a graph with injectable integrations for offline behavioral tests."""
    main = MAIN_PROMPT | factory.tools([info_tool, ToUnderwritingAssistant])
    underwriting = UNDERWRITING_PROMPT | factory.tools([info_tool, ToQuoteAssistant, CompleteOrEscalate])
    quote = QUOTE_PROMPT | factory.tools([fetch_rates, CompleteOrEscalate])
    rewrite = REWRITE_PROMPT | factory.chat() | StrOutputParser()
    reason = REASONING_PROMPT | factory.structured(CategoriesOutput)
    grade = GRADING_PROMPT | factory.structured(DescriptionGradeSchema)

    def entry(state: MainState) -> str:
        workflow = state.get("workflow_state", "main_assistant")
        return workflow if workflow in {"main_assistant", "underwriting_assistant", "quote_assistant"} else "main_assistant"

    def route(state: MainState) -> str:
        message = state["messages"][-1]
        calls = getattr(message, "tool_calls", [])
        if not calls:
            return END
        if len(calls) != 1:
            return "reject_tools"
        name = calls[0]["name"]
        workflow = entry(state)
        allowed = {
            "main_assistant": {"info_tool": "store_info", "ToUnderwritingAssistant": "begin_underwriting"},
            "underwriting_assistant": {"info_tool": "store_info", "ToQuoteAssistant": "begin_quote", "CompleteOrEscalate": "complete"},
            "quote_assistant": {"fetch_rates": "calculate_quote", "CompleteOrEscalate": "complete"},
        }
        return allowed[workflow].get(name, "reject_tools")

    def reject_tools(state: MainState) -> dict:
        return {"messages": [reply(call, "Invalid tool request. Call exactly ONE tool supported by your current role.") for call in state["messages"][-1].tool_calls]}

    def store_info(state: MainState) -> dict:
        call = state["messages"][-1].tool_calls[0]
        try:
            supplied = BusinessInformation.model_validate(call["args"]["business_info"])
            patch = supplied.model_dump(exclude_none=True, exclude_unset=True)
            if not patch or any(isinstance(v, str) and not v.strip() for v in patch.values()):
                raise ValueError("Business facts cannot be an empty update or blank strings.")
            info = {**state.get("business_information", {}), **patch}
            update = {"business_information": info, "messages": [reply(call, {"business_information": info})]}
            if any(state.get("business_information", {}).get(k) != v for k, v in patch.items()):
                update["last_quote"] = {}
            return update
        except (KeyError, TypeError, ValueError) as error:
            return {"messages": [reply(call, f"Invalid business update: {error}")]}

    def begin_underwriting(state: MainState) -> dict:
        call = state["messages"][-1].tool_calls[0]
        try:
            ToUnderwritingAssistant.model_validate(call["args"])
            missing = missing_fields(state.get("business_information", {}))
            if missing:
                raise ValueError(f"Ask the user for missing business fields first: {missing}")
            return {"tool_call_id": call["id"], "workflow_state": "underwriting_assistant", "last_quote": {}, "final_classifications": []}
        except (TypeError, ValueError) as error:
            return {"messages": [reply(call, str(error))]}

    def after_begin_underwriting(state: MainState) -> str:
        return "retrieve" if state.get("workflow_state") == "underwriting_assistant" else "main_assistant"

    def retrieve(state: MainState, config: RunnableConfig) -> dict:
        description = rewrite.invoke({"business_information": state["business_information"]}, config=config)
        documents = retriever.invoke(description, config=config)
        if not documents:
            raise ValueError("未检索到可用分类。")
        return {"description": description, "documents": documents}

    def reasoning(state: MainState, config: RunnableConfig) -> dict:
        output = reason.invoke({"description": state["description"], "documents": "\n".join(state["documents"])}, config=config)
        output = CategoriesOutput.model_validate(output)
        canonical, seen = [], set()
        for candidate in output.categories:
            record = catalog.match(candidate.category, candidate.code, candidate.cat_description)
            if record.text not in state["documents"]:
                raise ValueError("模型返回了未检索到的分类。")
            if record.text not in seen:
                seen.add(record.text)
                canonical.append({**candidate.model_dump(), "category": record.category, "cat_description": record.description, "rate": float(record.rate)})
        return {"rationale": canonical}

    def grading(state: MainState, config: RunnableConfig) -> dict:
        output = DescriptionGradeSchema.model_validate(grade.invoke({
            "description": state["description"], "rationale": json.dumps(state["rationale"], ensure_ascii=False),
        }, config=config))
        allowed = {(c["category"], c["code"], normalized(c["cat_description"])) for c in state["rationale"]}
        canonical, seen = [], set()
        for candidate in output.description_grades:
            record = catalog.match(candidate.category, candidate.code, candidate.cat_description)
            key = (record.category, record.code, normalized(record.description))
            if key not in allowed:
                raise ValueError("评分模型返回了未经分析的分类。")
            if key not in seen:
                seen.add(key)
                canonical.append({**candidate.model_dump(), "category": record.category, "cat_description": record.description})
        call = {"id": state["tool_call_id"], "name": "ToUnderwritingAssistant"}
        return {
            "graded_categories": canonical,
            "messages": [reply(call, {"instruction": "Ask the user to confirm these candidates; keep grades private.", "candidates": canonical})],
        }

    def begin_quote(state: MainState) -> dict:
        call = state["messages"][-1].tool_calls[0]
        try:
            args = ToQuoteAssistant.model_validate(call["args"])
            candidates = {(c["category"], c["code"], normalized(c["cat_description"])) for c in state["graded_categories"]}
            classifications, seen = [], set()
            for candidate in args.final_classifications:
                record = catalog.match(candidate.category, candidate.code, candidate.classification_description)
                key = (record.category, record.code, normalized(record.description))
                if key not in candidates:
                    raise ValueError("Use only the evaluated candidate classifications.")
                if key not in seen:
                    seen.add(key)
                    classifications.append({"category": record.category, "code": record.code, "classification_description": record.description})
            return {
                "final_classifications": classifications, "workflow_state": "quote_assistant", "last_quote": {},
                "messages": [reply(call, {"final_classifications": classifications, "instruction": "Calculate the tutorial quote."})],
            }
        except (KeyError, TypeError, ValueError) as error:
            return {"messages": [reply(call, f"Invalid classification handoff: {error}")]}

    def calculate_quote(state: MainState) -> dict:
        call = state["messages"][-1].tool_calls[0]
        try:
            request = RateRequest.model_validate(call["args"])
            requested = {name.strip().upper() for name in request.categories}
            confirmed = {c["category"] for c in state["final_classifications"]}
            if requested != confirmed:
                raise ValueError(f"Use all and only the confirmed categories: {sorted(confirmed)}")
            revenue = Decimal(str(state["business_information"]["projected_revenue"]))
            if Decimal(str(request.projected_revenue)) != revenue:
                raise ValueError("Use the stored projected_revenue without changing the amount.")
            existing = state.get("last_quote", {})
            result = existing or rates.calculate(sorted(confirmed), request.projected_revenue)
            return {"last_quote": result, "messages": [reply(call, result)]}
        except (KeyError, TypeError, ValueError) as error:
            return {"messages": [reply(call, f"Quote calculation rejected: {error}")]}

    def complete(state: MainState) -> dict:
        message = state["messages"][-1]
        call = message.tool_calls[0]
        try:
            args = CompleteOrEscalate.model_validate(call["args"])
            finished = state.get("workflow_state") == "quote_assistant" and args.cancel and bool(state.get("last_quote"))
            messages = [reply(call, "Quote complete." if finished else "Returned to the information assistant for corrections.")]
            if finished:
                messages.append(AIMessage(content=format_quote(state["last_quote"])))
            return {"workflow_state": "main_assistant", "messages": messages}
        except (TypeError, ValueError) as error:
            return {"messages": [reply(call, f"Invalid completion request: {error}")]}

    def after_complete(state: MainState) -> str:
        # Invalid completion arguments leave the role unchanged so it can repair them.
        if state.get("workflow_state") != "main_assistant":
            return entry(state)
        return END if state.get("last_quote") else "main_assistant"

    graph = StateGraph(MainState)
    for name, runnable in {
        "main_assistant": Assistant(main), "underwriting_assistant": Assistant(underwriting),
        "quote_assistant": Assistant(quote), "reject_tools": reject_tools,
        "store_info": store_info, "begin_underwriting": begin_underwriting,
        "retrieve": retrieve, "reasoning": reasoning, "grading": grading,
        "begin_quote": begin_quote, "calculate_quote": calculate_quote, "complete": complete,
    }.items():
        graph.add_node(name, runnable)
    graph.add_conditional_edges(START, entry)
    for name in ("main_assistant", "underwriting_assistant", "quote_assistant"):
        graph.add_conditional_edges(name, route)
    graph.add_conditional_edges("reject_tools", entry)
    graph.add_conditional_edges("store_info", entry)
    graph.add_conditional_edges("begin_underwriting", after_begin_underwriting)
    graph.add_edge("retrieve", "reasoning")
    graph.add_edge("reasoning", "grading")
    graph.add_edge("grading", "underwriting_assistant")
    graph.add_conditional_edges("begin_quote", entry)
    graph.add_edge("calculate_quote", "quote_assistant")
    graph.add_conditional_edges("complete", after_complete)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())
