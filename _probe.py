import inspect
from mcp import ClientSession, StdioServerParameters, stdio_client
print("stdio_client module:", stdio_client)
print("stdio_client file:", stdio_client.__file__ if hasattr(stdio_client,'__file__') else '?')

# stdio_client is a function returning (read_stream, write_stream)
sig = inspect.signature(stdio_client)
print("stdio_client sig:", sig)

print("=== ClientSession __init__ ===")
sig = inspect.signature(ClientSession.__init__)
print(sig)
print("=== ClientSession async methods ===")
print([x for x in dir(ClientSession) if not x.startswith("_") and not x[0].isupper()])

print("=== Tool ===")
from mcp.types import Tool
print(inspect.signature(Tool.__init__))
print("=== CallToolRequest ===")
from mcp.types import CallToolRequest
print("fields:", [x for x in dir(CallToolRequest) if not x.startswith("_")])
