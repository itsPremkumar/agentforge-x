import inspect
from langgraph.prebuilt import ToolNode
print("=== ToolNode init ===")
try:
    print(inspect.signature(ToolNode.__init__))
except Exception as e:
    print("err", e)
print("=== ToolNode methods ===")
print([x for x in dir(ToolNode) if not x.startswith("_")])
from langchain_core.tools import Tool as LCTool, InjectedToolCall
print("LC Tool fields ok")
# tool_node function
import langgraph.prebuilt as pre
print("prebuilt:", [x for x in dir(pre) if not x.startswith("_")])
