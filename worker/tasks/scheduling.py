"""Task Config: Meeting Scheduling & Constraint Solving."""

from typing import Any, Dict, Optional
from worker.core.llm import FinalAnswer
from worker.core.loop import TaskConfig
from worker.core.registry import ToolRegistry
from worker.core.verifier import VerifierResult
from worker.env.mock_calendar import MockCalendarEnvironment
from worker.tools.calendar import create_calendar_tools


def create_scheduling_task(
    calendar_env: Optional[MockCalendarEnvironment] = None,
) -> TaskConfig:
    env = calendar_env or MockCalendarEnvironment()
    tools = create_calendar_tools(env)

    def verify_meeting_scheduled(final_ans: FinalAnswer, registry: ToolRegistry) -> VerifierResult:
        """Independent verification read-back: checks event state directly in mock calendar."""
        data = final_ans.result_data or {}
        event_id = data.get("event_id")

        if not event_id:
            return VerifierResult(
                passed=False,
                reason="Verification failed: No 'event_id' provided in final answer result_data.",
            )

        event = env.get_event(event_id)
        if not event:
            return VerifierResult(
                passed=False,
                reason=f"Verification failed: Event '{event_id}' was not found in calendar store.",
            )

        if event.get("status") != "confirmed":
            return VerifierResult(
                passed=False,
                reason=f"Verification failed: Event status is '{event.get('status')}', expected 'confirmed'.",
            )

        if event.get("duration_min") != 45:
            return VerifierResult(
                passed=False,
                reason=f"Verification failed: Event duration is {event.get('duration_min')} mins, expected 45 mins.",
            )

        attendees = event.get("attendees", [])
        if len(attendees) < 2:
            return VerifierResult(
                passed=False,
                reason=f"Verification failed: Event attendees list {attendees} is missing required participants.",
            )

        return VerifierResult(
            passed=True,
            reason=f"Read-back verified: Meeting '{event_id}' confirmed with attendees {attendees} for 45 mins at {event.get('start')}.",
            details={"event": event},
        )

    return TaskConfig(
        name="Meeting Scheduling",
        goal="Set up a 45-minute call with Priya and Marcus next week to review the Q4 budget. Mornings are better. Avoid Wednesday.",
        tools=tools,
        system_hint=(
            "Use contacts.find to look up email addresses and timezones for Priya and Marcus. "
            "If contacts.find returns multiple matches for Marcus, ask the user to clarify which Marcus is intended. "
            "Use calendar.free_busy to inspect busy blocks for next week. "
            "Find a 45-minute slot that fits morning preferences and avoids Wednesday. "
            "Place a tentative hold using calendar.hold, then create the event with calendar.create_event."
        ),
        success_check=verify_meeting_scheduled,
        approval_rules={"require_approval_for_risk": ["irreversible"]},
        max_steps=20,
    )
