"""ERP tools for managing invoices, customer records, and financial ledgers."""

from typing import Any, Dict, List, Optional
from worker.core.registry import Tool
from worker.env.mock_erp import MockERPEnvironment


def create_erp_tools(erp_env: MockERPEnvironment) -> List[Tool]:
    """Factory function returning registered ERP Tool objects connected to erp_env."""

    def post_erp_invoice(
        invoice_number: str, vendor_id: str, amount: float, currency: str = "USD", due_date: str = ""
    ) -> Dict[str, Any]:
        return erp_env.post_invoice(
            invoice_number=invoice_number,
            vendor_id=vendor_id,
            amount=amount,
            currency=currency,
            due_date=due_date,
        )

    def get_erp_invoice(invoice_number: str) -> Optional[Dict[str, Any]]:
        res = erp_env.get_invoice(invoice_number=invoice_number)
        if not res:
            return {"found": False, "message": f"Invoice '{invoice_number}' not found in ERP database."}
        return {"found": True, "invoice": res}

    def search_customer(query: str) -> List[Dict[str, Any]]:
        return erp_env.search_customer_by_name(query=query)

    def get_customer(customer_id: str) -> Optional[Dict[str, Any]]:
        res = erp_env.get_customer(customer_id=customer_id)
        if not res:
            return {"found": False, "message": f"Customer '{customer_id}' not found in ERP."}
        return {"found": True, "customer": res}

    def update_customer_record(customer_id: str, field_name: str, new_value: str) -> Dict[str, Any]:
        return erp_env.update_customer_field(
            customer_id=customer_id, field_name=field_name, new_value=new_value
        )

    def get_erp_expenses() -> List[Dict[str, Any]]:
        return erp_env.get_erp_expenses()

    return [
        Tool(
            name="post_erp_invoice",
            description="Post a new invoice into the ERP accounting system.",
            params_schema={
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string", "description": "Unique invoice identifier."},
                    "vendor_id": {"type": "string", "description": "Vendor account ID."},
                    "amount": {"type": "number", "description": "Total monetary amount."},
                    "currency": {"type": "string", "default": "USD"},
                    "due_date": {"type": "string", "description": "ISO format due date YYYY-MM-DD."},
                },
                "required": ["invoice_number", "vendor_id", "amount"],
            },
            fn=post_erp_invoice,
            risk="write",
            verify_hint="Read back posted invoice using get_erp_invoice to verify state.",
        ),
        Tool(
            name="get_erp_invoice",
            description="Query and retrieve an invoice record from ERP by invoice number.",
            params_schema={
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string", "description": "Invoice number to lookup."}
                },
                "required": ["invoice_number"],
            },
            fn=get_erp_invoice,
            risk="read",
            verify_hint="Returns invoice details if registered.",
        ),
        Tool(
            name="search_customer",
            description="Search customer database by name or ID snippet.",
            params_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Name or ID snippet to search."}
                },
                "required": ["query"],
            },
            fn=search_customer,
            risk="read",
        ),
        Tool(
            name="get_customer",
            description="Retrieve detailed record for a specific customer ID.",
            params_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Unique customer ID."}
                },
                "required": ["customer_id"],
            },
            fn=get_customer,
            risk="read",
        ),
        Tool(
            name="update_customer_record",
            description="Update a specific field on a customer record in ERP.",
            params_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Target customer ID."},
                    "field_name": {
                        "type": "string",
                        "description": "Field name to update (e.g., email, phone, tier).",
                    },
                    "new_value": {"type": "string", "description": "New value to write."},
                },
                "required": ["customer_id", "field_name", "new_value"],
            },
            fn=update_customer_record,
            risk="irreversible",
            verify_hint="Read back customer details with get_customer to verify update.",
        ),
        Tool(
            name="get_erp_expenses",
            description="Fetch all recorded expense entries from ERP for reconciliation.",
            params_schema={
                "type": "object",
                "properties": {},
            },
            fn=get_erp_expenses,
            risk="read",
        ),
    ]
