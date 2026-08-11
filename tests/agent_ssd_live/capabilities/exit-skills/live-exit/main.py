from agent_wiki.backend.core.capabilities.tool import sign_tool


@sign_tool(short_description="Return the literal live-test result to the parent.")
def main(
    message: str,
    selected_tool_outputs: list[str | None],
) -> dict[str, object]:
    """Return the standard exit payload used by the runtime validator."""
    return {
        "message": message,
        "selected_tool_outputs": selected_tool_outputs,
    }
