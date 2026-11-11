"""Search server for live questions.

Optional MCP server that returns web search results for live forecasting
questions. Retrieved text is data, never instructions.
"""

from datetime import UTC, datetime
from typing import Literal


def search_web(
    query: str,
    as_of: str,
    source: Literal["metaculus", "good_judgment_open", "general"] = "general",
) -> dict:
    """Search for live questions matching the query.

    Returns synthetic search results with source URL and retrieval date.
    In production, this would call the Metaculus API, Good Judgment Open API,
    or a web search engine.
    """
    now = datetime.now(UTC)
    # Synthetic results - in production, this would query live APIs.
    results = [
        {
            "query": query,
            "results": [
                {
                    "title": f"Simulated result for {query}",
                    "url": f"https://example.com/simulated/{hash(query) % 10000}",
                    "source": source,
                    "retrieval_date": now.isoformat(),
                    "snippet": "Simulated search snippet. In production, this returns "
                    "live question text from Metaculus, Good Judgment Open, "
                    "or a web search engine.",
                }
            ],
            "total_results": 1,
            "search_time_seconds": 0.5,
        }
    ]
    return {
        "query": query,
        "as_of": as_of,
        "results": results,
        "generated_at": now.isoformat(),
    }


def create_search_server():
    """Create the search MCP server."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        name="forecast-search",
        instructions="Search for live forecasting questions from web sources.",
        json_response=True,
    )

    @server.tool()
    def search_web_tool(
        ctx,
        query: str,
        source: str = "general",
        as_of: str | None = None,
    ) -> dict:
        if as_of is None:
            as_of = datetime.now(UTC).isoformat()
        return search_web(query, as_of=as_of, source=source)

    return server


def main():
    create_search_server().run()


if __name__ == "__main__":
    main()
