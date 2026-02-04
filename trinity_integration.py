
import sys
import os
from fastmcp import FastMCP, Context

# Add Trinity AI path
trinity_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trinity-ai-damage-assessment", "src"))
if trinity_path not in sys.path:
    sys.path.append(trinity_path)

try:
    from trinity_damage_assessment.assessment import DamageAssessor
except ImportError:
    # Fallback if path setup fails or file doesn't exist yet
    class DamageAssessor:
        def assess_image(self, image_path, asset_type="roof"):
            return {"status": "error", "message": "Trinity AI module not found"}

from finance_trading_ai_agents_mcp.addition_custom_mcp.addition_custom_mcp_interface import AdditionCustomMcpInterface

class TrinityCustomMcp(AdditionCustomMcpInterface):
    """
    Integrates Trinity AI Damage Assessment into the OrPaynter MCP ecosystem.
    """
    
    def __init__(self):
        self.assessor = DamageAssessor()
        super().__init__()
        
    def add_mcp_server_name(self) -> FastMCP:
        """
        Registers the Trinity AI MCP server.
        """
        mcp = FastMCP("trinity_ai_intelligence")
        
        @mcp.tool(title="Assess Property Damage")
        async def assess_damage(context: Context, image_path: str, asset_type: str = "roof") -> str:
            """
            Analyzes an image of property (roof, siding, etc.) to detect damage and estimate repair costs.
            
            Args:
                image_path: Path to the image file.
                asset_type: Type of asset ('roof', 'siding', 'window', 'gutter').
            """
            result = self.assessor.assess_image(image_path, asset_type)
            return str(result)
            
        return mcp

    # Implement required abstract methods with pass (since we only need one server)
    def custom_economic_calendar_mcp(self): pass
    def custom_news_mcp(self): pass
    def custom_price_action_mcp(self): pass
    def custom_traditional_indicator_mcp(self): pass
    
    def add_mcp_server_name1(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name2(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name3(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name4(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name5(self) -> FastMCP: raise NotImplementedError
