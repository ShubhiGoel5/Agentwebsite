"""Mock ERP system environment with database state and fault injection support."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ERPInvoice:
    invoice_number: str
    vendor_id: str
    amount: float
    currency: str = "USD"
    status: str = "POSTED"
    due_date: str = ""
    posted_at: str = "2026-10-04"


@dataclass
class ERPCustomer:
    customer_id: str
    name: str
    email: str
    phone: str
    tier: str = "Standard"
    status: str = "Active"


class MockERPEnvironment:
    """In-memory ERP system backend with support for fault injection."""

    def __init__(self):
        self.invoices: Dict[str, ERPInvoice] = {}
        self.customers: Dict[str, ERPCustomer] = {}
        self.expenses: List[Dict[str, Any]] = []

        # Fault injection controls
        self.transient_error_counter: int = 0  # Number of next calls to throw 503
        self.reject_duplicates: bool = True

        self._seed_default_data()

    def _seed_default_data(self):
        # Initial customers
        self.customers["CUST-101"] = ERPCustomer(
            customer_id="CUST-101",
            name="Apex Technologies",
            email="billing@apextech.com",
            phone="+1-555-0199",
            tier="Enterprise",
        )
        self.customers["CUST-102"] = ERPCustomer(
            customer_id="CUST-102",
            name="Starlight Media",
            email="info@starlightmedia.org",
            phone="+1-555-0244",
            tier="Standard",
        )

        # Initial ERP recorded expenses for reconciliation test
        self.expenses = [
            {"expense_id": "EXP-001", "date": "2026-09-01", "description": "Software Subscription", "amount": 150.00, "category": "Software"},
            {"expense_id": "EXP-002", "date": "2026-09-05", "description": "Team Lunch", "amount": 85.50, "category": "Meals"},
            {"expense_id": "EXP-003", "date": "2026-09-12", "description": "Client Flight", "amount": 420.00, "category": "Travel"},
            {"expense_id": "EXP-004", "date": "2026-09-18", "description": "Office Stationeries", "amount": 65.00, "category": "Office"},
        ]

    def _check_fault_injection(self):
        if self.transient_error_counter > 0:
            self.transient_error_counter -= 1
            raise ConnectionError("503 Service Unavailable: ERP database temporarily busy (Fault Injection).")

    # --- Invoice Methods ---
    def post_invoice(self, invoice_number: str, vendor_id: str, amount: float, currency: str = "USD", due_date: str = "") -> Dict[str, Any]:
        self._check_fault_injection()
        if self.reject_duplicates and invoice_number in self.invoices:
            raise ValueError(f"Duplicate Invoice Error: Invoice '{invoice_number}' already exists in ERP.")

        inv = ERPInvoice(
            invoice_number=invoice_number,
            vendor_id=vendor_id,
            amount=float(amount),
            currency=currency,
            due_date=due_date,
        )
        self.invoices[invoice_number] = inv
        return {
            "status": "success",
            "message": f"Invoice '{invoice_number}' successfully posted.",
            "invoice": inv.__dict__,
        }

    def get_invoice(self, invoice_number: str) -> Optional[Dict[str, Any]]:
        self._check_fault_injection()
        inv = self.invoices.get(invoice_number)
        return inv.__dict__ if inv else None

    # --- Customer Methods ---
    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        self._check_fault_injection()
        cust = self.customers.get(customer_id)
        return cust.__dict__ if cust else None

    def search_customer_by_name(self, query: str) -> List[Dict[str, Any]]:
        self._check_fault_injection()
        results = []
        for c in self.customers.values():
            if query.lower() in c.name.lower() or query.lower() in c.customer_id.lower():
                results.append(c.__dict__)
        return results

    def update_customer_field(self, customer_id: str, field_name: str, new_value: Any) -> Dict[str, Any]:
        self._check_fault_injection()
        if customer_id not in self.customers:
            raise KeyError(f"Customer '{customer_id}' not found in ERP.")

        cust = self.customers[customer_id]
        if not hasattr(cust, field_name):
            raise AttributeError(f"Customer schema does not contain field '{field_name}'. Valid fields: {list(cust.__dict__.keys())}")

        setattr(cust, field_name, new_value)
        return {
            "status": "success",
            "customer_id": customer_id,
            "field_name": field_name,
            "updated_value": new_value,
            "customer": cust.__dict__,
        }

    # --- Expense Reconciliation Methods ---
    def list_erp_expenses(self) -> List[Dict[str, Any]]:
        return self.expenses

    def get_erp_expenses(self) -> List[Dict[str, Any]]:
        self._check_fault_injection()
        return self.expenses
