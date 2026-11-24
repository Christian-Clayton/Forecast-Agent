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

    # Browser fetch tool for live web content
    @server.tool()
    def browser_fetch(
        ctx,
        url: str,
        max_chars: int = 50000,
    ) -> dict:
        """Fetch and return the text content of a live web page.

        Uses Playwright to navigate to a URL and extract readable text.
        The page is loaded with JavaScript enabled. Results are truncated
        to max_chars to avoid token limits.

        NOTE: This tool uses threading because the MCP call_tool is
        synchronous and cannot await async code. The browser fetch runs
        in a separate thread.
        """
        import asyncio
        import threading
        from datetime import UTC, datetime

        results = {
            "url": url,
            "fetched_at": datetime.now(UTC).isoformat(),
            "content": None,
            "char_count": 0,
            "error": None,
        }

        async def _fetch_async():
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    content = await page.inner_text("body")
                    content = " ".join(content.split())
                    results["content"] = content[:max_chars]
                    results["char_count"] = len(content)
                except Exception as e:
                    results["error"] = str(e)
                finally:
                    await browser.close()

        # Run the async function in a separate thread using asyncio.run()
        def _run_in_thread():
            nonlocal results
            try:
                asyncio.run(_fetch_async())
            except Exception as e:
                results["error"] = f"Browser fetch failed: {e}"

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=60)  # Wait up to 60s for the fetch to complete

        return results

    return server


def main():
    create_search_server().run()


if __name__ == "__main__":
    main()
