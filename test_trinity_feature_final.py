
import asyncio
import json
import sys
import os

# Setup Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "aitrados_api"))

from trinity_integration import TrinityCustomMcp

async def test_trinity_feature():
    print("--- Testing TrinityCustomMcp Functionality ---")
    
    # 1. Initialize
    mcp_integration = TrinityCustomMcp()
    
    # 2. Get Server
    mcp_server = mcp_integration.add_mcp_server_name()
    print(f"✅ Server created: {mcp_server.name}")

    # 3. Inspect and Run Tool
    if hasattr(mcp_server, '_tools'):
        print(f"Found _tools: {len(mcp_server._tools)} items")
        for t in mcp_server._tools:
            if "Assess Property Damage" in t.name:
                print(f"✅ Found Tool: {t.name}")
                
                # Mock context
                class MockContext:
                    request_id = "test-123"
                
                print("   -> Executing tool...")
                try:
                    res = await t.fn(MockContext(), image_path="test.jpg", asset_type="roof")
                    result_data = json.loads(res)
                    
                    if result_data.get("status") == "success" and result_data.get("asset_type") == "roof":
                         print("✅ Tool execution successful!")
                         print(f"   Severity: {result_data['findings']['severity']}")
                         print(f"   Cost: ${result_data['findings']['estimated_repair_cost_usd']}")
                    else:
                         print("❌ Tool execution result unexpected:")
                         print(res)
                except Exception as e:
                    print(f"   -> Execution error: {e}")
                return

    print("⚠️ Tool not found in registry (check FastMCP version implementation).")

if __name__ == "__main__":
    asyncio.run(test_trinity_feature())
