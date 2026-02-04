
import asyncio
import sys
import os
import inspect

# Setup Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "aitrados_api"))

from trinity_integration import TrinityCustomMcp

async def debug_mcp_tools():
    print("--- Debugging TrinityCustomMcp Tools ---")
    
    mcp_integration = TrinityCustomMcp()
    mcp_server = mcp_integration.add_mcp_server_name()
    
    print(f"Server Name: {mcp_server.name}")
    
    # Try public API first
    try:
        tools = await mcp_server.list_tools() # standard mcp protocol method usually?
        # Wait, FastMCP wraps it.
        # Looking at dir(mcp), we saw 'get_tools' (sync or async?)
    except:
        pass

    # Try get_tools() which might return the internal dict or list
    try:
        print("Calling get_tools()...")
        # In some versions get_tools is async, in others sync.
        # It's likely returning a list of Tool objects
        tools = mcp_server.get_tools()
        if inspect.iscoroutine(tools):
             tools = await tools
        
        print(f"get_tools() returned type: {type(tools)}")
        
        if isinstance(tools, list):
            print(f"Count: {len(tools)}")
            for t in tools:
                print(f" - {t}")
                if hasattr(t, 'name'): print(f"   Name: {t.name}")
    except Exception as e:
        print(f"get_tools() failed: {e}")

    # Try accessing _tool_manager
    if hasattr(mcp_server, '_tool_manager'):
        tm = mcp_server._tool_manager
        print(f"_tool_manager type: {type(tm)}")
        # It might have .tools or ._tools
        if hasattr(tm, '_tools'):
             print(f"_tool_manager._tools keys: {tm._tools.keys()}")
             
             # Try to execute the tool if found
             if "Assess Property Damage" in tm._tools:
                 print("✅ Found tool in _tool_manager!")
                 tool_wrapper = tm._tools["Assess Property Damage"]
                 # tool_wrapper might be the function itself or a Tool object
                 print(f"Tool wrapper type: {type(tool_wrapper)}")
                 
                 # Invoke it
                 class MockContext:
                    request_id = "debug-123"
                    
                 try:
                     # Usually the tool wrapper is callable or has a run method
                     # FastMCP 2.x often wraps it in a Tool object with a .fn or .run
                     if hasattr(tool_wrapper, 'fn'):
                         res = await tool_wrapper.fn(MockContext(), image_path="test.jpg", asset_type="roof")
                         print(f"Execution Result: {res[:50]}...")
                     elif callable(tool_wrapper):
                         res = await tool_wrapper(MockContext(), image_path="test.jpg", asset_type="roof")
                         print(f"Execution Result: {res[:50]}...")
                 except Exception as e:
                     print(f"Execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_mcp_tools())
