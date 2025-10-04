"""Entry point for the Context Switcher MCP server."""

import asyncio
import logging
import os
import sys

from . import create_server

# Configure logging to stderr to avoid interfering with MCP protocol
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the MCP server with stdio transport (default)."""
    logger.info("Starting Context Switcher MCP server (stdio)...")
    try:
        server = create_server()
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("Server error")
        raise


def http_main(host: str = "127.0.0.1", port: int = 8082) -> None:
    """Run the MCP server with HTTP transport using FastMCP's native support.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 8082)
    """
    logger.info(f"Starting Context Switcher MCP server (HTTP) on {host}:{port}")
    try:
        # Create server with HTTP settings
        server = create_server(host=host, port=port)
        # Use run_streamable_http_async for HTTP transport
        asyncio.run(server.run_streamable_http_async())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("Server error")
        raise


if __name__ == "__main__":
    # Check if HTTP mode requested
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8082"))
        http_main(host=host, port=port)
    else:
        main()
