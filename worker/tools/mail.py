"""Mail tools for interacting with the Mail service."""

from typing import Any, Dict, List, Optional
from worker.core.registry import Tool
from worker.env.mock_mail import MockMailEnvironment


def create_mail_tools(mail_env: MockMailEnvironment) -> List[Tool]:
    """Factory function returning registered mail Tool objects connected to mail_env."""

    def list_emails(unread_only: bool = False) -> List[Dict[str, Any]]:
        return mail_env.list_emails(unread_only=unread_only)

    def read_email(email_id: str) -> Dict[str, Any]:
        return mail_env.get_email_details(email_id=email_id)

    tool_list = [
        Tool(
            name="list_emails",
            description="List emails in the inbox. Optionally filter by unread status.",
            params_schema={
                "type": "object",
                "properties": {
                    "unread_only": {
                        "type": "boolean",
                        "description": "If true, only returns unread emails.",
                        "default": False,
                    }
                },
            },
            fn=list_emails,
            risk="read",
            verify_hint="Lists messages available in mailbox.",
        ),
        Tool(
            name="read_email",
            description="Retrieve full details, body text, and attachments for a specific email ID.",
            params_schema={
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "Unique identifier of the email."}
                },
                "required": ["email_id"],
            },
            fn=read_email,
            risk="read",
            verify_hint="Fetches email content and attachment text.",
        ),
    ]
    return tool_list
