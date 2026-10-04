"""Mock Calendar & Contacts Environment for constraint solving, race condition, and scheduling tests."""

from typing import Any, Dict, List, Optional, Union


class MockCalendarEnvironment:
    """In-memory mock calendar service with race condition simulation and contact search."""

    def __init__(self, simulate_race_condition: bool = False):
        self.simulate_race_condition = simulate_race_condition
        self.race_condition_triggered = False

        # Pre-seeded contacts directory
        self.contacts = [
            {"name": "Priya Sharma", "email": "priya.sharma@acme.com", "timezone": "America/New_York"},
            {"name": "Marcus Vance", "email": "marcus.vance@acme.com", "timezone": "America/Los_Angeles"},
            {"name": "Marcus Miller", "email": "marcus.miller@acme.com", "timezone": "America/New_York"},
            {"name": "Alex Taylor", "email": "alex.taylor@acme.com", "timezone": "America/Chicago"},
        ]

        # Pre-seeded busy slots for next week (2026-10-12 Mon to 2026-10-16 Fri)
        self.busy_slots = {
            "priya.sharma@acme.com": [
                {"start": "2026-10-12T09:00:00", "end": "2026-10-12T10:00:00"},
                {"start": "2026-10-14T09:00:00", "end": "2026-10-14T17:00:00"},  # All day Wednesday
                {"start": "2026-10-15T11:00:00", "end": "2026-10-15T12:00:00"},
            ],
            "marcus.vance@acme.com": [
                {"start": "2026-10-12T10:00:00", "end": "2026-10-12T11:30:00"},
                {"start": "2026-10-13T14:00:00", "end": "2026-10-13T16:00:00"},
                {"start": "2026-10-14T08:00:00", "end": "2026-10-14T18:00:00"},  # Wednesday busy
            ],
            "marcus.miller@acme.com": [
                {"start": "2026-10-13T09:00:00", "end": "2026-10-13T10:00:00"},
                {"start": "2026-10-15T09:00:00", "end": "2026-10-15T10:00:00"},
            ],
        }

        self.holds: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}

    def find_contacts(self, name: str) -> Dict[str, Any]:
        """Finds contacts by partial name match. Returns matches."""
        query = name.strip().lower()
        matches = [c for c in self.contacts if query in c["name"].lower() or query in c["email"].lower()]
        
        if len(matches) == 1:
            return {"status": "success", "count": 1, "matches": matches, "contact": matches[0]}
        elif len(matches) > 1:
            return {
                "status": "ambiguous",
                "count": len(matches),
                "message": f"Multiple contacts match '{name}'. Please clarify which person is intended.",
                "matches": matches,
            }
        else:
            return {"status": "not_found", "count": 0, "matches": [], "message": f"No contact found for '{name}'."}

    def free_busy(self, emails: List[str], start_date: str = "2026-10-12", end_date: str = "2026-10-16") -> Dict[str, Any]:
        """Returns busy schedule blocks for given emails."""
        result = {}
        for email in emails:
            result[email] = self.busy_slots.get(email, [])
        return {
            "query": {"emails": emails, "start_date": start_date, "end_date": end_date},
            "busy_blocks": result,
            "working_hours": "09:00 to 17:00 local time",
        }

    def hold_slot(self, slot_start: str, slot_end: str, attendees: List[str]) -> Dict[str, Any]:
        """Attempts to place a tentative hold on a slot. Simulates race condition if enabled."""
        slot_key = f"{slot_start}_{slot_end}"

        if self.simulate_race_condition and not self.race_condition_triggered:
            self.race_condition_triggered = True
            # Simulate another user snatching the slot right before hold
            for att in attendees:
                if att not in self.busy_slots:
                    self.busy_slots[att] = []
                self.busy_slots[att].append({"start": slot_start, "end": slot_end})
            return {
                "status": "conflict_race_condition",
                "success": False,
                "message": f"Slot {slot_start} was just booked by another user. Please choose another slot.",
            }

        # Check existing holds / conflicts
        if slot_key in self.holds:
            return {
                "status": "conflict",
                "success": False,
                "message": f"Slot {slot_start} is already held or reserved.",
            }

        hold_id = f"hold_{len(self.holds) + 1:03d}"
        hold_record = {
            "hold_id": hold_id,
            "slot_start": slot_start,
            "slot_end": slot_end,
            "attendees": attendees,
            "status": "active",
        }
        self.holds[slot_key] = hold_record
        return {"status": "success", "success": True, "hold_id": hold_id, "hold": hold_record}

    def create_event(
        self,
        slot_start: str,
        slot_end: str,
        attendees: List[str],
        title: str = "Q4 Budget Review",
        description: str = "",
        duration_min: int = 45,
    ) -> Dict[str, Any]:
        """Creates an event and sends calendar invitations (Irreversible action)."""
        event_id = f"evt_{len(self.events) + 101}"
        event_record = {
            "event_id": event_id,
            "title": title,
            "description": description,
            "start": slot_start,
            "end": slot_end,
            "duration_min": duration_min,
            "attendees": attendees,
            "status": "confirmed",
        }
        self.events[event_id] = event_record
        return {"status": "success", "event_id": event_id, "event": event_record}

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Returns event details for independent verification."""
        return self.events.get(event_id)

    def cancel_event(self, event_id: str) -> Dict[str, Any]:
        """Cancels a scheduled event."""
        if event_id in self.events:
            self.events[event_id]["status"] = "cancelled"
            return {"status": "success", "message": f"Event '{event_id}' has been cancelled."}
        return {"status": "error", "message": f"Event '{event_id}' not found."}
