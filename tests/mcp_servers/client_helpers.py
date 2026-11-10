import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

ResultT = TypeVar("ResultT")


def run_client(
    server: FastMCP,
    operation: Callable[[object], Awaitable[ResultT]],
) -> ResultT:
    async def execute() -> ResultT:
        async with create_connected_server_and_client_session(server) as session:
            return await operation(session)

    return asyncio.run(execute())


def result_payload(result: object) -> object:
    if result.isError:
        raise AssertionError(result.content[0].text)
    if result.structuredContent is not None:
        if isinstance(result.structuredContent, dict) and {
            "tool_call_id",
            "result",
        } <= set(result.structuredContent):
            return result.structuredContent["result"]
        if isinstance(result.structuredContent, dict) and set(
            result.structuredContent
        ) == {"result"}:
            return result.structuredContent["result"]
        return result.structuredContent
    return json.loads(result.content[0].text)
