
import asyncio
import json
import sys
import os
import inspect

# Setup Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "aitrados_api"))

from trinity_integration import TrinityCustomMcp

async def verify_trinity_integration():
    print("================================================")
    print("   Trinity AI Damage Assessment - Integration Verification")
    print("================================================")
    
    # 1. Initialize Integration
    try:
        print("\n1. Initializing TrinityCustomMcp Module...")
        mcp_integration = TrinityCustomMcp()
        print("   ✅ Initialization successful.")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return

    # 2. Register Server
    try:
        print("\n2. Registering MCP Server...")
        mcp_server = mcp_integration.add_mcp_server_name()
        print(f"   ✅ Server registered: '{mcp_server.name}'")
    except Exception as e:
        print(f"   ❌ Registration failed: {e}")
        return

    # 3. Locate Tool
    print("\n3. Verifying Tool Registration...")
    tool_name = "assess_damage"
    tool_found = False
    tool_func = None
    
    # Check via get_tools()
    # FastMCP 2.x often makes get_tools async
    tools_result = mcp_server.get_tools()
    if inspect.iscoroutine(tools_result):
        tools = await tools_result
    else:
        tools = tools_result
        
    if tool_name in tools:
        print(f"   ✅ Tool '{tool_name}' found in registry.")
        tool_found = True
        tool_entry = tools[tool_name]
        
        # FastMCP 2.x: tool_entry is usually a Tool object with a .fn attribute
        if hasattr(tool_entry, 'fn'):
            tool_func = tool_entry.fn
        elif callable(tool_entry):
            tool_func = tool_entry
    else:
        print(f"   ⚠️ Tool '{tool_name}' not in get_tools() keys: {list(tools.keys())}")
        return

    # 4. Execute Tool (Mocked)
    if tool_func:
        print("\n4. Executing Damage Assessment (Mock)...")
        print("   Input: image='test_roof.jpg', asset='roof'")
        
        class MockContext:
            request_id = "verify-123"
            
        try:
            # Execute
            result_json = await tool_func(MockContext(), image_path="test_roof.jpg", asset_type="roof")
            
            # Parse Result
            result = json.loads(result_json)
            
            print("\n   --- Assessment Result ---")
            print(f"   ID:       {result.get('assessment_id')}")
            print(f"   Status:   {result.get('status')}")
            
            if result.get('status') != 'failed':
                findings = result.get('findings', {})
                print(f"   Severity: {findings.get('severity')}")
                print(f"   Issues:   {findings.get('detected_issues')}")
                print(f"   Cost:     ${findings.get('estimated_repair_cost_usd')}")
                print("   -------------------------")
                print("   ✅ Verification PASSED: End-to-end execution successful.")
            else:
                print(f"   ❌ Verification FAILED: AI returned error status: {result.get('error')}")

        except Exception as e:
            print(f"   ❌ Execution Exception: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ❌ Verification FAILED: Could not resolve callable function for tool.")

if __name__ == "__main__":
    asyncio.run(verify_trinity_integration())
