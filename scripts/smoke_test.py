"""Opt-in live check. This calls paid APIs using only the synthetic GABAS example."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quoting_agent.app import Conversation, create_graph
from quoting_agent.config import Settings
from quoting_agent.llm import ModelFactory
from quoting_agent.models import DescriptionGradeSchema
from quoting_agent.pricing import format_quote


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    settings = Settings.load(args.env_file)
    factory = ModelFactory(settings)
    # Probe the primary directly: a successful fallback must not hide incompatibility.
    probe = factory.models[0].with_structured_output(DescriptionGradeSchema, method="function_calling")
    result = probe.invoke(
        "Return a description_grades list with exactly one item: category CONSULTANTS, "
        "cat_description Computer Consultants--Application Service Provider., "
        "code 68673, grade 0.95."
    )
    if not isinstance(result, DescriptionGradeSchema):
        raise AssertionError("Primary provider returned no validated structured result")
    print("主模型结构化输出检查通过。", flush=True)

    conversation = Conversation(create_graph(settings), settings.recursion_limit)
    inputs = [
        "GABAS, LLC provides tailor-made AI solutions as hosted internet applications. "
        "We design and build software systems end-to-end. Secondary operations are AI strategy "
        "and IT consulting. We operate in the technology industry as software service providers, "
        "not physical goods distributors. We work mostly remotely with occasional on-site visits. "
        "Projected annual revenue is USD 4,000,000. Please summarize and begin underwriting.",
        "I confirm that summary. We host AI applications, build software systems end-to-end, "
        "and provide AI strategy and IT consulting. We work primarily remotely with occasional "
        "on-site visits, without manufacturing, hardware sales or physical goods distribution. "
        "Please proceed with underwriting and classification confirmation.",
        "I confirm the applicable software application service provider, software system "
        "development and IT consulting classifications you identified, including remote work. "
        "There are no additional business operations. Please finalize the applicable "
        "classifications and calculate the sample quote on USD 4,000,000 annual revenue.",
        "The relevant classifications you listed are confirmed. All supplied facts and the "
        "USD 4,000,000 revenue are correct. Please proceed with the sample quotation.",
    ]
    for index, text in enumerate(inputs, 1):
        responses = conversation.ask(text)
        print(f"第 {index} 轮完成；状态：{conversation.state.get('workflow_state', 'main_assistant')}；回复数：{len(responses)}", flush=True)
        if conversation.state.get("last_quote"):
            break
    state = conversation.state
    if not state.get("last_quote"):
        raise AssertionError("No quote after four scripted turns; inspect prompts and classifications")
    messages = state["messages"]
    requested = [c["id"] for m in messages for c in getattr(m, "tool_calls", [])]
    answered = [m.tool_call_id for m in messages if m.type == "tool"]
    if sorted(requested) != sorted(answered):
        raise AssertionError("Unanswered or duplicate tool results")
    report = {
        "primary_structured_output": "passed", "live_graph": "passed", "turns": index,
        "tool_protocol": "passed", "quote": state["last_quote"],
        "workflow_state": state["workflow_state"],
    }
    settings.runtime_path.mkdir(parents=True, exist_ok=True)
    (settings.runtime_path / "live_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_quote(state["last_quote"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
