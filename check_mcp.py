#!/usr/bin/env python3
"""Check MCPServer attribute names."""
from mcp.server import MCPServer
import inspect

# Create a test server
s = MCPServer(name="test-server", description="test")
print("Public attrs:", [a for a in dir(s) if not a.startswith('_')])
print("Name attr:", getattr(s, 'name', 'N/A'))
print("Title attr:", getattr(s, 'title', 'N/A'))

# Check what add_tool stores
def my_tool(x: int) -> dict:
    """A test tool."""
    return {"result": x}

s.add_tool(my_tool, name="my_tool", description="test tool")
print("After add_tool:")
print("Tool list:", [t for t in dir(s) if not t.startswith('_')])
