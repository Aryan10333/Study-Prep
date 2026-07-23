# Simple stdio MCP Server
import asyncio
from mcp.server.models import InitializationOptions
from mcp.server import Server
import mcp.types as types
from mcp.server.stdio import stdio_server

server = Server("user-db-server")

@server.list_tools()
async def handle_list_tools():
    return [
        types.Tool(
            name="lookup_user",
            description="Look up user records by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "lookup_user":
        username = arguments.get("name")
        return [types.TextContent(type="text", text=f"User database record: {username}, Role=AI_Engineer, Salary=$150,000")]
    raise ValueError(f"Tool {name} not found")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="user-db-server",
                server_version="1.0.0",
                capabilities=types.ServerCapabilities(
                    tools=types.ToolsCapability()
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
