"""LangChain adapter для локального FastMCP-сервера."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)

from guardrails import TOOL_ALLOWLIST


BASE_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = BASE_DIR / "mcp_server.py"


def build_mcp_client() -> MultiServerMCPClient:
    """Створити MCP client для локального stdio server."""

    return MultiServerMCPClient(
        {
            "requirements_estimation": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(MCP_SERVER_PATH)],
            }
        }
    )


async def load_all_mcp_tools() -> list[BaseTool]:
    """Завантажити всі tools через MCP protocol."""

    client = build_mcp_client()
    tools = await client.get_tools()

    return list(tools)


async def load_tools_for_agent(
    agent_name: str,
) -> list[BaseTool]:
    """Надати agent лише tools з його allowlist."""

    if agent_name not in TOOL_ALLOWLIST:
        raise ValueError(
            f"Unknown agent: {agent_name}"
        )

    allowed_names = TOOL_ALLOWLIST[agent_name]
    all_tools = await load_all_mcp_tools()

    return [
        tool
        for tool in all_tools
        if tool.name in allowed_names
    ]


def index_tools(
    tools: list[BaseTool],
) -> dict[str, BaseTool]:
    """Побудувати registry MCP tools за іменем."""

    return {
        tool.name: tool
        for tool in tools
    }


async def call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Викликати MCP tool через LangChain adapter."""

    tools = index_tools(
        await load_all_mcp_tools()
    )

    if tool_name not in tools:
        raise ValueError(
            f"Unknown MCP tool: {tool_name}"
        )

    return await tools[tool_name].ainvoke(
        arguments
    )


async def main() -> None:
    """Продемонструвати MCP integration."""

    tools = await load_all_mcp_tools()

    print("LangChain MCP tools:")

    for tool in tools:
        print(f"- {tool.name}")

    result = await call_mcp_tool(
        "check_requirements_readiness",
        {
            "initiative_id": "DEM-010",
            "business_objective": (
                "Скоротити demand lead time."
            ),
        },
    )

    print("\nMCP adapter result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())