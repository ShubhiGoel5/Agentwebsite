"""Mock Email service environment for simulated email fetching and attachment extraction."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Email:
    id: str
    sender: str
    subject: str
    body: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    read: bool = False


class MockMailEnvironment:
    """In-memory Mail server simulator."""

    def __init__(self):
        self.inbox: Dict[str, Email] = {}
        self._seed_default_data()

    def _seed_default_data(self):
        self.inbox["msg_001"] = Email(
            id="msg_001",
            sender="vendor@acmesupplies.com",
            subject="Invoice #INV-2026-9912 for Acme Corp",
            body="Dear Accounts Payable,\n\nPlease find attached invoice #INV-2026-9912 for January supplies.\nTotal Amount: $4,250.00\nDue Date: 2026-10-31\nVendor Tax ID: TAX-9912-US",
            attachments=[
                {
                    "filename": "INV-2026-9912.pdf",
                    "content_type": "application/pdf",
                    "raw_text": "INVOICE NUMBER: INV-2026-9912\nVENDOR: Acme Supplies Inc.\nVENDOR_ID: VEND-ACME-01\nAMOUNT: 4250.00\nCURRENCY: USD\nITEMS: Office Furniture (10), Printers (2)\nDUE_DATE: 2026-10-31",
                }
            ],
        )
        self.inbox["msg_002"] = Email(
            id="msg_002",
            sender="support@cloudservices.io",
            subject="Monthly Cloud Hosting Invoice #INV-CLOUD-881",
            body="Your monthly server usage for September total is $890.15.",
            attachments=[],
        )

    def list_unread_emails(self) -> List[Dict[str, Any]]:
        return self.list_emails(unread_only=True)

    def list_emails(self, unread_only: bool = False) -> List[Dict[str, Any]]:
        results = []
        for em in self.inbox.values():
            if unread_only and em.read:
                continue
            results.append({
                "id": em.id,
                "sender": em.sender,
                "subject": em.subject,
                "read": em.read,
                "has_attachments": len(em.attachments) > 0,
            })
        return results

    def get_email_details(self, email_id: str) -> Dict[str, Any]:
        if email_id not in self.inbox:
            raise KeyError(f"Email with ID '{email_id}' not found.")
        em = self.inbox[email_id]
        em.read = True
        return {
            "id": em.id,
            "sender": em.sender,
            "subject": em.subject,
            "body": em.body,
            "attachments": em.attachments,
        }
