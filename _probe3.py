import inspect
from langchain_core.tools import Tool as LCTool, BaseTool
print("=== LC Tool init ===")
print(inspect.signature(LCTool.__init__))
print("=== BaseTool fields ===")
print([x for x in dir(BaseTool) if not x.startswith("_")][:30])
print("=== LC Tool methods of interest ===")
for m in ["invoke","ainvoke","bind","as_tool"]:
    if hasattr(LCTool, m):
        print(m, inspect.signature(getattr(LCTool, m)))
from langchain_core.tools.base import ToolInput
print("ToolInput ok")
# Check tool_node
from langgraph.prebuilt import tool_node
print("tool_node:", [x for x in dir(tool_node) if not x.startswith("_")][:20])
