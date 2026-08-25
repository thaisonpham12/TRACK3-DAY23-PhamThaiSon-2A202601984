"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Structured Output Schema for Classification ────────────────────
class ClassificationResult(BaseModel):
    """Structured intent classification output."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Classification route. Priority: risky > tool > missing_info > error > simple"
    )
    risk_level: Literal["high", "low"] = Field(
        description="'high' for risky actions (refunds, deletions), 'low' otherwise"
    )


# ─── Node Implementations ───────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    Uses .with_structured_output() with Pydantic schema for reliable classification.
    Routes: simple, tool, missing_info, risky, error.
    Priority: risky > tool > missing_info > error > simple.
    """
    query = state.get("query", "").strip()

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(ClassificationResult)
        prompt = (
            "You are an intent classifier for a customer support AI system.\n"
            "Classify the following query into exactly one route:\n"
            "- 'risky': Actions requiring human approval, refunds, account deletion, "
            "security/permission changes.\n"
            "- 'tool': Queries requiring external tool execution, lookup (e.g. order status, "
            "database lookup, fetching records).\n"
            "- 'missing_info': Ambiguous or incomplete queries lacking details to act "
            "(e.g. 'Can you fix it?').\n"
            "- 'error': Reports of transient failures, timeouts, crashes requiring retry.\n"
            "- 'simple': Direct informational questions, FAQs, password reset guides.\n\n"
            "Priority ordering (if ambiguous): risky > tool > missing_info > error > simple.\n"
            "For risk_level: set to 'high' if route is 'risky', else 'low'.\n\n"
            f"User Query: {query}"
        )
        decision = structured_llm.invoke(prompt)
        route: str
        risk_level: str
        if isinstance(decision, ClassificationResult):
            route = str(decision.route)
            risk_level = str(decision.risk_level)
        elif isinstance(decision, dict):
            route = str(decision.get("route", "simple"))
            risk_level = str(decision.get("risk_level", "high" if route == "risky" else "low"))
        else:
            route = str(getattr(decision, "route", "simple"))
            risk_level = str(getattr(decision, "risk_level", "high" if route == "risky" else "low"))

        return {
            "route": route,
            "risk_level": risk_level,
            "events": [
                make_event(
                    "classify",
                    "completed",
                    f"classified as {route}",
                    route=route,
                    risk_level=risk_level,
                )
            ],
        }
    except Exception as exc:
        # Fallback policy: record failure in errors/events and fallback safely
        fallback_route = "simple"
        fallback_risk = "low"
        return {
            "route": fallback_route,
            "risk_level": fallback_risk,
            "errors": [f"LLM classification failed: {exc}"],
            "events": [
                make_event(
                    "classify",
                    "failed",
                    f"LLM classification error: {exc}",
                    fallback_route=fallback_route,
                )
            ],
        }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    proposed_action = state.get("proposed_action")

    if route == "error" and attempt < 2:
        result = f"ERROR: Transient timeout/failure for query: '{query}' (attempt {attempt})"
        event = make_event("tool", "failed", "mock tool returned error", attempt=attempt)
    else:
        target = proposed_action or query
        result = f"SUCCESS: Tool executed successfully for '{target}'"
        event = make_event("tool", "completed", "mock tool completed", attempt=attempt)

    return {
        "tool_results": [result],
        "events": [event],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.
    """
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    if "ERROR" in latest_result or not latest_result:
        verdict = "needs_retry"
    else:
        verdict = "success"

    return {
        "evaluation_result": verdict,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation verdict: {verdict}",
                evaluation_result=verdict,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response grounded in context using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")

    try:
        llm = get_llm(temperature=0.0)
        context_parts = [f"User Query: {query}"]
        if tool_results:
            context_parts.append(f"Tool Results: {' | '.join(tool_results)}")
        if proposed_action:
            context_parts.append(f"Proposed Action: {proposed_action}")
        if approval:
            status = "Approved" if approval.get("approved") else "Rejected"
            reviewer = approval.get("reviewer", "reviewer")
            context_parts.append(f"Approval Decision: {status} (Reviewed by {reviewer})")

        prompt = (
            "You are a helpful and accurate customer support AI assistant.\n"
            "Generate a clear, grounded, and concise final response to the user query "
            "based ONLY on the provided context.\n\n"
            + "\n".join(context_parts)
            + "\n\nFinal Response:"
        )
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return {
            "final_answer": str(answer_text).strip(),
            "events": [make_event("answer", "completed", "grounded final answer generated")],
        }
    except Exception as exc:
        fallback_answer = f"Response for '{query}': Processed successfully."
        if tool_results:
            fallback_answer += f" Result: {tool_results[-1]}"
        return {
            "final_answer": fallback_answer,
            "errors": [f"Answer generation LLM error: {exc}"],
            "events": [make_event("answer", "failed", f"LLM error: {exc}")],
        }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")

    if approval and not approval.get("approved", True):
        comment = approval.get("comment", "Action was not approved")
        target = proposed_action or query
        question = (
            f"The requested action '{target}' was rejected ({comment}). "
            "Could you please clarify your requirements or provide an alternative request?"
        )
    else:
        question = (
            f"Could you please provide more specific details or context regarding: '{query}'?"
        )

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            make_event(
                "clarify",
                "completed",
                "clarification question generated",
                pending_question=question,
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    action = f"Execute action for request: '{query}'"

    return {
        "proposed_action": action,
        "events": [
            make_event(
                "risky_action",
                "completed",
                "action proposed for review",
                proposed_action=action,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    """
    decision = {
        "approved": True,
        "reviewer": "security-reviewer",
        "comment": "Approved by security policy",
    }

    return {
        "approval": decision,
        "events": [
            make_event(
                "approval",
                "completed",
                "approval decision recorded",
                approved=decision["approved"],
                reviewer=decision["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.
    """
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1
    max_attempts = state.get("max_attempts", 3)

    return {
        "attempt": new_attempt,
        "errors": [f"Attempt {new_attempt}/{max_attempts} failed with transient error."],
        "events": [
            make_event(
                "retry",
                "completed",
                f"retry recorded attempt {new_attempt}",
                attempt=new_attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    query = state.get("query", "")
    attempt = state.get("attempt", 0)
    answer = (
        f"Unable to complete request '{query}' after {attempt} retry attempts. "
        "The issue has been logged and escalated to support engineering."
    )

    return {
        "final_answer": answer,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "exhausted max retries, escalated to dead letter",
                attempt=attempt,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
