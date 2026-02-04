
import asyncio
import json
import sys
import os

# Setup Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir) # For finance_trading_ai_agents_mcp
sys.path.append(os.path.join(base_dir, "aitrados_api"))

from trinity_integration import TrinityCustomMcp

async def test_trinity_feature():
    print("--- Testing TrinityCustomMcp ---")
    
    # 1. Initialize
    try:
        mcp_integration = TrinityCustomMcp()
        print("✅ Initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Check inheritance
    from finance_trading_ai_agents_mcp.addition_custom_mcp.addition_custom_mcp_interface import AdditionCustomMcpInterface
    if isinstance(mcp_integration, AdditionCustomMcpInterface):
         print("✅ Inheritance verification passed")
    else:
         print("⚠️ Inheritance verification failed (using dummy base?)")

    # 2. Get Server
    try:
        mcp_server = mcp_integration.add_mcp_server_name()
        print(f"✅ Server created: {mcp_server.name}")
    except Exception as e:
        print(f"❌ Server creation failed: {e}")
        return

    # 3. Inspect and Run Tool
    # FastMCP typically exposes tools via list_tools() which returns models, or we can look at internal storage.
    # We will try to invoke the method directly by creating a wrapper since we know the method name is defined inside add_mcp_server_name.
    # But we can't access the inner function from outside easily.
    
    # However, we can use the mcp_server to call the tool if FastMCP supports local execution.
    # FastMCP.call_tool(name, arguments) might exist?
    
    print(f"Server attributes: {dir(mcp_server)}")
    
    # Attempt to find the tool function
    found = False
    # Check if _tools is a list of Tool objects
    if hasattr(mcp_server, '_tools'):
        print(f"Found _tools: {len(mcp_server._tools)} items")
        for t in mcp_server._tools:
            print(f" - Tool: {t.name}")
            if "Assess Property Damage" in t.name:
                found = True
                print("   -> Executing tool...")
                # Mock context
                class MockContext:
                    request_id = "test-123"
                
                try:
                    # t.fn is usually the function
                    res = await t.fn(MockContext(), image_path="test.jpg", asset_type="roof")
                    print(f"   -> Result: {res[:100]}...")
                except Exception as e:
                    print(f"   -> Execution error: {e}")
                
    if not found:
        print("⚠️ Tool not found in _tools registry.")

if __name__ == "__main__":
    asyncio.run(test_trinity_feature())
