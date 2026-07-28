import asyncio
from mcp.server import Server
import mcp.types as types
from mcp.server.stdio import stdio_server

server = Server("user-db-server")

@server.list_tools()
async def handle_list_tools():
    return [
        types.Tool(
            name="lookup_user",
            description="Look up user profile fields.",
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
        username = arguments.get("name", "").lower()
        if "alice" in username:
            return [types.TextContent(type="text", text="Profile: Alice, Role: Administrator, Status: Active")]
        return [types.TextContent(type="text", text=f"Profile: {username}, Role: Guest, Status: Pending")]
    raise ValueError("Unknown tool")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())