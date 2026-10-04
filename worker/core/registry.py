"""Tool dataclass and registry for dynamic tool execution & schema validation."""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


RiskLevel = Literal["read", "write", "irreversible"]


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    is_transient: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_observation_str(self) -> str:
        if self.success:
            if isinstance(self.output, (dict, list)):
                import json
                return json.dumps(self.output, indent=2)
            return str(self.output)
        else:
            err_type = "Transient Error" if self.is_transient else "Tool Execution Error"
            return f"[{err_type}] {self.error or 'Unknown error occurred.'}"


@dataclass
class Tool:
    name: str
    description: str
    params_schema: Dict[str, Any]
    fn: Callable[..., Any]
    risk: RiskLevel = "read"
    verify_hint: Optional[str] = None

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        """Validates arguments against JSON schema. Returns error string if invalid."""
        if not isinstance(args, dict):
            return f"Arguments must be a JSON object (dict), got {type(args).__name__}"

        if HAS_JSONSCHEMA and self.params_schema:
            try:
                jsonschema.validate(instance=args, schema=self.params_schema)
            except jsonschema.ValidationError as e:
                return f"Schema validation error for tool '{self.name}': {e.message} at path '{'.'.join(str(p) for p in e.path)}'"
            except jsonschema.SchemaError as e:
                return f"Internal tool schema definition error for '{self.name}': {e.message}"
        else:
            # Basic fallback validation for required fields
            required = self.params_schema.get("required", [])
            for req in required:
                if req not in args:
                    return f"Missing required parameter '{req}' for tool '{self.name}'"
        return None

    def to_openai_schema(self) -> Dict[str, Any]:
        """Converts tool definition to OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_schema,
            },
        }


class ToolRegistry:
    """Registry for managing and executing tools safely."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Executes a tool with validation, exception handling, and error formatting."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found in registry. Available tools: {list(self._tools.keys())}",
                is_transient=False,
            )

        # 1. Schema Validation
        val_error = tool.validate_args(args)
        if val_error:
            return ToolResult(
                success=False,
                output=None,
                error=val_error,
                is_transient=False,
            )

        # 2. Tool Execution with error boundary
        try:
            # Support both kwarg unpacking and dictionary if function expects dict
            sig = inspect.signature(tool.fn)
            if len(sig.parameters) == 1 and list(sig.parameters.keys())[0] in ("args", "params", "data"):
                res = tool.fn(args)
            else:
                res = tool.fn(**args)
                
            return ToolResult(success=True, output=res)
        except ConnectionError as e:
            return ToolResult(
                success=False, output=None, error=f"Transient network error: {str(e)}", is_transient=True
            )
        except TimeoutError as e:
            return ToolResult(
                success=False, output=None, error=f"Tool execution timed out: {str(e)}", is_transient=True
            )
        except Exception as e:
            # Check for custom transient flag on exception if present
            is_transient = getattr(e, "is_transient", False)
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error in '{name}': {type(e).__name__} - {str(e)}",
                is_transient=is_transient,
            )
