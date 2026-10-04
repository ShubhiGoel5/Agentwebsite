"""Calendar and Contacts tools definitions."""

from typing import List
from worker.core.registry import Tool
from worker.env.mock_calendar import MockCalendarEnvironment


def create_calendar_tools(calendar_env: MockCalendarEnvironment) -> List[Tool]:
    """Creates tool wrappers for calendar and contacts functionality."""

    contacts_find_tool = Tool(
        name="contacts.find",
        description="Search contacts by name to retrieve email addresses and timezones. Returns multiple matches if ambiguous.",
        params_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full or partial contact name to search."}
            },
            "required": ["name"],
        },
        fn=calendar_env.find_contacts,
        risk="read",
    )

    free_busy_tool = Tool(
        name="calendar.free_busy",
        description="Query calendar busy blocks for a list of attendee emails over a specified date range.",
        params_schema={
            "type": "object",
            "properties": {
                "emails": {"type": "array", "items": {"type": "string"}, "description": "List of attendee email addresses."},
                "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format (e.g. 2026-10-12)."},
                "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format (e.g. 2026-10-16)."},
            },
            "required": ["emails"],
        },
        fn=calendar_env.free_busy,
        risk="read",
    )

    hold_tool = Tool(
        name="calendar.hold",
        description="Place a tentative hold on a time slot for specified attendees before final confirmation.",
        params_schema={
            "type": "object",
            "properties": {
                "slot_start": {"type": "string", "description": "Start time in ISO format (e.g. 2026-10-13T10:00:00)."},
                "slot_end": {"type": "string", "description": "End time in ISO format (e.g. 2026-10-13T10:45:00)."},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee email addresses."},
            },
            "required": ["slot_start", "slot_end", "attendees"],
        },
        fn=calendar_env.hold_slot,
        risk="write",
    )

    create_event_tool = Tool(
        name="calendar.create_event",
        description="Create a confirmed calendar event and send invitations to attendees (Irreversible action).",
        params_schema={
            "type": "object",
            "properties": {
                "slot_start": {"type": "string", "description": "Start time in ISO format (e.g. 2026-10-13T10:00:00)."},
                "slot_end": {"type": "string", "description": "End time in ISO format (e.g. 2026-10-13T10:45:00)."},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee email addresses."},
                "title": {"type": "string", "description": "Title of the meeting."},
                "description": {"type": "string", "description": "Optional description/agenda."},
                "duration_min": {"type": "integer", "description": "Duration in minutes (default 45)."},
            },
            "required": ["slot_start", "slot_end", "attendees"],
        },
        fn=calendar_env.create_event,
        risk="irreversible",
    )

    get_event_tool = Tool(
        name="calendar.get_event",
        description="Fetch details of a scheduled event by event_id for verification.",
        params_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "The unique event ID to fetch."}
            },
            "required": ["event_id"],
        },
        fn=calendar_env.get_event,
        risk="read",
    )

    cancel_tool = Tool(
        name="calendar.cancel",
        description="Cancel a scheduled calendar event by event_id.",
        params_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "The event ID to cancel."}
            },
            "required": ["event_id"],
        },
        fn=calendar_env.cancel_event,
        risk="write",
    )

    return [
        contacts_find_tool,
        free_busy_tool,
        hold_tool,
        create_event_tool,
        get_event_tool,
        cancel_tool,
    ]
