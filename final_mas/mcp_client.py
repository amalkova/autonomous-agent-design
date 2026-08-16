"""LangChain adapter для локального FastMCP-сервера."""

from __future__ import annotations

import asyncio
import json
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


def normalize_mcp_result(
    result: Any,
) -> Any:
    """Перетворити MCP adapter result на JSON-safe data."""

    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result

    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                text = item.get("text")

                if text is not None:
                    return normalize_mcp_result(text)

            text = getattr(item, "text", None)

            if text is not None:
                return normalize_mcp_result(text)

        return [
            normalize_mcp_result(item)
            for item in result
        ]

    if isinstance(result, dict):
        return {
            str(key): normalize_mcp_result(value)
            for key, value in result.items()
        }

    return result


async def run_integration_demos() -> dict[str, Any]:
    """Виконати два demo через MultiServerMCPClient."""

    tools = index_tools(
        await load_all_mcp_tools()
    )

    readiness = await tools[
        "check_requirements_readiness"
    ].ainvoke(
        {
            "initiative_id": "DEM-910",
            "business_objective": (
                "Скоротити demand lead time."
            ),
        }
    )

    complexity = await tools[
        "classify_estimation_complexity"
    ].ainvoke(
        {
            "initiative_id": "DEM-911",
            "systems_count": 3,
            "integration_count": 2,
            "nfr_criticality": "high",
            "data_migration_required": True,
            "security_review_required": True,
            "dependency_count": 2,
            "requirements_stability": "partial",
        }
    )

    return {
        "client": "MultiServerMCPClient",
        "transport": "stdio",
        "server": "requirements_estimation",
        "available_tools": sorted(tools),
        "demos": [
            {
                "scenario": "requirements_readiness",
                "result": normalize_mcp_result(
                    readiness
                ),
            },
            {
                "scenario": "complexity_estimation",
                "result": normalize_mcp_result(
                    complexity
                ),
            },
        ],
    }


async def main() -> None:
    """Запустити та зберегти MCP integration demos."""

    artifact = await run_integration_demos()

    output_path = (
        BASE_DIR
        / "artifacts"
        / "mcp_integration_demo.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"\\nSaved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
