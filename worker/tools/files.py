"""File manipulation tools for reading CSV data and extracting document contents."""

import csv
import io
from typing import Any, Dict, List, Optional
from worker.core.registry import Tool


def create_file_tools() -> List[Tool]:
    """Factory returning file reader tools."""

    def read_csv_file(csv_content: str) -> List[Dict[str, Any]]:
        """Parses CSV string into a list of row dictionaries."""
        f = io.StringIO(csv_content.strip())
        reader = csv.DictReader(f)
        return [row for row in reader]

    def parse_pdf_text(raw_text: str) -> Dict[str, Any]:
        """Extracts structured key-value lines from PDF text dump."""
        lines = raw_text.strip().split("\n")
        extracted = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                extracted[k.strip().lower()] = v.strip()
        return {"raw_text": raw_text, "extracted_fields": extracted}

    return [
        Tool(
            name="read_csv_file",
            description="Parse raw CSV content string into structured JSON rows.",
            params_schema={
                "type": "object",
                "properties": {
                    "csv_content": {"type": "string", "description": "Raw string content of CSV file."}
                },
                "required": ["csv_content"],
            },
            fn=read_csv_file,
            risk="read",
        ),
        Tool(
            name="parse_pdf_text",
            description="Extract text content and key-value fields from PDF text.",
            params_schema={
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string", "description": "Raw text content extracted from PDF."}
                },
                "required": ["raw_text"],
            },
            fn=parse_pdf_text,
            risk="read",
        ),
    ]
