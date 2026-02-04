
import sys
import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger
from fastmcp import FastMCP, Context

# Add Trinity AI path
trinity_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trinity-ai-damage-assessment", "src"))
if trinity_path not in sys.path:
    sys.path.append(trinity_path)

# Try to import the Trinity AI module
try:
    from trinity_damage_assessment.assessment import DamageAssessor
    TRINITY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Trinity AI module: {e}")
    TRINITY_AVAILABLE = False
    # Fallback class to prevent crash during class definition
    class DamageAssessor:
        def assess_image(self, *args, **kwargs):
            raise ImportError("Trinity AI module not available")

# Import the base interface (now supported by our mock aitrados_api)
try:
    from finance_trading_ai_agents_mcp.addition_custom_mcp.addition_custom_mcp_interface import AdditionCustomMcpInterface
except ImportError:
    # If the import still fails for some reason, define a dummy base class
    logger.warning("AdditionCustomMcpInterface not found, using dummy base.")
    class AdditionCustomMcpInterface:
        def __init__(self): pass

# --- Data Models ---

class DamageFindings(BaseModel):
    severity: str = Field(..., description="Severity level of the damage (none, low, medium, high, critical)")
    confidence_score: float = Field(..., description="AI confidence score between 0.0 and 1.0")
    detected_issues: List[str] = Field(..., description="List of specific damage types identified")
    estimated_repair_cost_usd: float = Field(..., description="Estimated cost of repair in USD")

class AssessmentResult(BaseModel):
    assessment_id: str = Field(..., description="Unique ID for the assessment")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the assessment")
    asset_type: str = Field(..., description="Type of asset analyzed (roof, siding, etc.)")
    image_analyzed: str = Field(..., description="Path or identifier of the image analyzed")
    findings: DamageFindings = Field(..., description="Detailed findings of the assessment")
    recommendation: str = Field(..., description="Actionable recommendation based on severity")
    status: str = Field("success", description="Status of the operation")
    error: Optional[str] = Field(None, description="Error message if failed")

# --- Main Class ---

class TrinityCustomMcp(AdditionCustomMcpInterface):
    """
    Integrates Trinity AI Damage Assessment into the OrPaynter MCP ecosystem.
    Provides tools for property damage analysis and repair cost estimation.
    """
    
    def __init__(self):
        logger.info("Initializing TrinityCustomMcp integration...")
        self.assessor = DamageAssessor() if TRINITY_AVAILABLE else None
        
        # Configuration (could be loaded from env)
        self.api_key = os.getenv("TRINITY_API_KEY", "mock_key")
        self.endpoint_url = os.getenv("TRINITY_ENDPOINT", "local_module")
        
        super().__init__()
        logger.success("TrinityCustomMcp initialized.")

    def add_mcp_server_name(self) -> FastMCP:
        """
        Registers the Trinity AI MCP server and its tools.
        """
        mcp = FastMCP("trinity_ai_intelligence")
        
        @mcp.tool(title="Assess Property Damage")
        async def assess_damage(context: Context, image_path: str, asset_type: str = "roof") -> str:
            """
            Analyzes an image of property (roof, siding, window, gutter) to detect damage.
            
            Use this tool when you need to:
            - Check a property photo for storm damage
            - Estimate repair costs for insurance claims
            - Verify the condition of a roof or siding
            
            Args:
                image_path: Absolute path to the image file or URL.
                asset_type: Type of asset to analyze. Supported: 'roof', 'siding', 'window', 'gutter'.
            
            Returns:
                JSON string containing the assessment results, severity, and cost estimates.
            """
            request_id = context.request_id if hasattr(context, 'request_id') else "unknown"
            logger.info(f"[{request_id}] Received damage assessment request for {asset_type} at {image_path}")
            
            if not self.assessor:
                logger.error("Trinity AI module is not initialized.")
                return json.dumps({
                    "status": "failed",
                    "error": "Trinity AI service unavailable"
                })

            try:
                # Validate Input
                valid_assets = ["roof", "siding", "window", "gutter"]
                if asset_type.lower() not in valid_assets:
                    logger.warning(f"Invalid asset type requested: {asset_type}")
                    return json.dumps({
                        "status": "failed",
                        "error": f"Invalid asset_type. Must be one of {valid_assets}"
                    })

                # Simulate AI Processing
                logger.debug(f"Invoking Trinity AI DamageAssessor for {image_path}...")
                
                # In a real async scenario, we might await this if it was an async call
                # For this mock/synchronous module, we call it directly
                raw_result = self.assessor.assess_image(image_path, asset_type)
                
                if raw_result.get("status") == "failed":
                     logger.error(f"AI Assessment failed: {raw_result.get('error')}")
                     return json.dumps(raw_result)

                # Process Result
                logger.success(f"Assessment complete. ID: {raw_result.get('assessment_id')}")
                
                # Return formatted JSON
                # We return a string because FastMCP handles basic types best, 
                # and complex Pydantic returns can sometimes be tricky depending on the client.
                # However, returning a JSON string is universally safe.
                return json.dumps(raw_result, indent=2)

            except Exception as e:
                logger.exception(f"Unexpected error during damage assessment: {e}")
                return json.dumps({
                    "status": "error",
                    "error": f"Internal server error: {str(e)}"
                })
            
        return mcp

    # Implement required abstract methods with pass (since we only use add_mcp_server_name)
    def custom_economic_calendar_mcp(self): pass
    def custom_news_mcp(self): pass
    def custom_price_action_mcp(self): pass
    def custom_traditional_indicator_mcp(self): pass
    
    def add_mcp_server_name1(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name2(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name3(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name4(self) -> FastMCP: raise NotImplementedError
    def add_mcp_server_name5(self) -> FastMCP: raise NotImplementedError
