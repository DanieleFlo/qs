from agent_wiki.backend.core.capabilities.tool import sign_tool


@sign_tool(short_description="Return the fixed B test value.")
def main() -> str:
    """Return one deterministic value."""
    return "FIXED_TOOL_B_OK"
