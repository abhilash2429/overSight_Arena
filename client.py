# client.py
"""
Oversight Arena — OpenEnv client.

Usage (sync):
    from client import OversightArenaClient

    with OversightArenaClient("http://localhost:7860").sync() as env:
        result = env.reset(difficulty="easy", seed=42)
        obs_text = result.metadata["pipeline_text"]

        from openenv.core.env_server.mcp_types import CallToolAction
        result = env.step(CallToolAction(
            tool_name="observe_worker",
            arguments={"worker_id": 3}
        ))
        print(result.observation.metadata["pipeline_text"])
        print(result.reward)

Usage (async):
    from client import OversightArenaClient
    import asyncio

    async def main():
        async with OversightArenaClient("http://localhost:7860") as env:
            result = await env.reset(difficulty="easy", seed=42)
            result = await env.call_tool("observe_worker", worker_id=3)

    asyncio.run(main())
"""

from openenv.core.mcp_client import MCPToolClient


class OversightArenaClient(MCPToolClient):
    """
    Client for the Oversight Arena environment.

    Inherits full MCPToolClient functionality:
      - reset(**kwargs)         → Observation
      - step(CallToolAction)    → StepResult
      - call_tool(name, **kw)   → tool result
      - list_tools()            → List[Tool]
      - state()                 → State
      - close() / async context manager

    Helper: parse_action_text(text) converts <action>...</action> strings
    to the appropriate CallToolAction for use in RL training loops.
    """

    pass


# Convenience alias used by training notebooks
OpenEnvClient = OversightArenaClient


def parse_action_text(action_text: str):
    """
    Parse LLM-generated action text into a CallToolAction.

    Converts the <action>VERB N [instruction]</action> format produced by
    the supervisor model into a typed CallToolAction for env.step().

    Supported formats:
      <action>OBSERVE 3</action>
      <action>DEEP_INSPECT 2</action>
      <action>REDIRECT 1 Focus on the first paragraph only.</action>
      <action>TERMINATE 4</action>
      <action>APPROVE 5</action>

    Returns CallToolAction on success, or None if parsing fails.
    """
    import re

    from openenv.core.env_server.mcp_types import CallToolAction

    _VERB_TO_TOOL = {
        "OBSERVE": "observe_worker",
        "DEEP_INSPECT": "deep_inspect_worker",
        "REDIRECT": "redirect_worker",
        "TERMINATE": "terminate_worker",
        "APPROVE": "approve_worker",
    }

    match = re.search(r"<action>(.*?)</action>", action_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    content = match.group(1).strip()
    parts = content.split(None, 2)  # split on whitespace, max 3 parts
    if len(parts) < 2:
        return None

    verb = parts[0].upper()
    tool_name = _VERB_TO_TOOL.get(verb)
    if tool_name is None:
        return None

    try:
        worker_id = int(parts[1])
    except ValueError:
        return None

    if not 1 <= worker_id <= 5:
        return None

    arguments = {"worker_id": worker_id}
    if verb == "REDIRECT":
        arguments["instruction"] = parts[2] if len(parts) > 2 else ""

    return CallToolAction(tool_name=tool_name, arguments=arguments)
