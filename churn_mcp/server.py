"""MCP server: the churn platform's tools for any MCP client (Claude Desktop, Claude Code, IDEs).

The tools are the same functions the in-app agent uses (service/agent_tools.py); MCP builds
their schemas from the same signatures and docstrings. They call the churn API over HTTP, so
this process needs only `mcp` - no model, no data.

  pip install "mcp>=2.3"
  CHURN_API_URL=http://localhost:8000 python -m churn_mcp.server                 # stdio (Claude Desktop/Code)
  CHURN_API_URL=http://localhost:8000 python -m churn_mcp.server --http         # streamable HTTP on :8765/mcp

Claude Code:  claude mcp add churn-platform -e CHURN_API_URL=http://localhost:8000 -- \\
                uv run --directory /path/to/saas-churn-platform --with "mcp>=2.3" python -m churn_mcp.server
"""
import functools
import sys

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from service import agent_tools

INSTRUCTIONS = (
    "Tools for a churn-prediction platform: score and look up customers, check data drift, "
    "classify consumer complaints and find similar past cases, and see what the automated "
    "workflows did. All tools are read-only. Customer data is a demo dataset (IBM Telco)."
)


def _expected_errors_to_client(fn):
    """MCP hides unexpected exceptions from the client ("Error executing tool"); a ToolError's
    message is passed on, so the model learns why a call failed (bad ID, API down) and can
    recover. functools.wraps keeps the signature, which MCP turns into the input schema."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except agent_tools.ToolError as e:
            raise ToolError(str(e)) from e
    return wrapper


server = MCPServer("churn-platform", instructions=INSTRUCTIONS)
for fn in agent_tools.TOOLS:
    server.add_tool(_expected_errors_to_client(fn),
                    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))


def main() -> None:
    if "--http" in sys.argv:
        server.run("streamable-http", host="127.0.0.1", port=8765)
    else:
        server.run()  # stdio: the client launches this process and talks over stdin/stdout


if __name__ == "__main__":
    main()
