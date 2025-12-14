"""
OrPaynter Financial MCP Server

Comprehensive MCP server for OrPaynter's automated financial monitoring and cash flow optimization.

This extends the aitrados financial MCP server with:
- Cash Flow Optimization Engine
- Risk Management System  
- Predictive Analytics
- Alert System
- Enhanced Market Data Integration
"""

import os
import json
from typing import Optional, Dict, Any
import asyncio
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastmcp import FastMCP
from finance_trading_ai_agents_mcp.mcp_manage import mcp_run as base_mcp_run
from finance_trading_ai_agents_mcp.mcp_services.global_instance import McpGlobalVar


class OrPaynterMCPConfig:
    """Configuration for OrPaynter Financial MCP Server"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.name = "OrPaynter Financial Monitoring MCP"
        self.description = "Comprehensive financial monitoring and cash flow optimization"
        
        # Default API keys (to be set by user)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.polygon_key = os.getenv('POLYGON_KEY')
        self.yahoo_finance_enabled = True
        
        # Risk management defaults
        self.default_risk_tolerance = float(os.getenv('DEFAULT_RISK_TOLERANCE', '0.15'))  # 15%
        self.max_portfolio_allocation = float(os.getenv('MAX_PORTFOLIO_ALLOCATION', '0.25'))  # 25%
        self.default_stop_loss = float(os.getenv('DEFAULT_STOP_LOSS', '0.10'))  # 10%
        self.default_take_profit = float(os.getenv('DEFAULT_TAKE_PROFIT', '0.20'))  # 20%
        
        # Alert settings
        self.price_alert_threshold = float(os.getenv('PRICE_ALERT_THRESHOLD', '0.05'))  # 5%
        self.volume_alert_multiplier = float(os.getenv('VOLUME_ALERT_MULTIPLIER', '2.0'))  # 2x
        self.volatility_alert_threshold = float(os.getenv('VOLATILITY_ALERT_THRESHOLD', '0.30'))  # 30%
        
        # Cash flow optimization
        self.auto_rebalance_enabled = os.getenv('AUTO_REBALANCE_ENABLED', 'true').lower() == 'true'
        self.dividend_reinvestment_enabled = os.getenv('DIVIDEND_REINVESTMENT_ENABLED', 'true').lower() == 'true'
        self.tax_loss_harvesting_enabled = os.getenv('TAX_LOSS_HARVESTING_ENABLED', 'true').lower() == 'true'
        
        # Market data settings
        self.market_data_update_interval = int(os.getenv('MARKET_DATA_UPDATE_INTERVAL', '60'))  # seconds
        self.historical_data_days = int(os.getenv('HISTORICAL_DATA_DAYS', '365'))  # days
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))


def validate_environment():
    """Validate required environment variables"""
    required_vars = ['AITRADOS_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == 'YOUR_SECRET_KEY':
            missing_vars.append(var)
    
    if missing_vars:
        logging.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logging.warning("Please set AITRADOS_SECRET_KEY in your environment")
    
    return len(missing_vars) == 0


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('orpaynter_finance_mcp.log'),
            logging.StreamHandler()
        ]
    )


def load_custom_config():
    """Load custom configuration from environment or config file"""
    config = OrPaynterMCPConfig()
    
    # Load additional config from environment
    config_files = ['config.json', '.env', 'orpaynter_config.json']
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        custom_config = json.load(f)
                    else:
                        # Parse .env file
                        custom_config = {}
                        for line in f:
                            if '=' in line and not line.strip().startswith('#'):
                                key, value = line.strip().split('=', 1)
                                custom_config[key] = value
                    
                    # Update config with custom values
                    for key, value in custom_config.items():
                        if hasattr(config, key.lower()):
                            setattr(config, key.lower(), value)
                break
            except Exception as e:
                logging.warning(f"Failed to load config file {config_file}: {e}")
    
    return config


def mcp_run(
    port: int = 11999, 
    host: str = "127.0.0.1", 
    addition_custom_mcp_py_file: Optional[str] = None,
    config_file: Optional[str] = None
):
    """
    Start OrPaynter Financial MCP Server
    
    Args:
        port: Server port
        host: Server host
        addition_custom_mcp_py_file: Custom MCP file path
        config_file: Configuration file path
    """
    # Setup
    setup_logging()
    config = load_custom_config()
    validate_environment()
    
    # Log startup information
    logging.info("Starting OrPaynter Financial MCP Server v%s", config.version)
    logging.info("Host: %s, Port: %d", host, port)
    
    # Load global configurations
    from aitrados_api.common_lib.common import load_global_configs
    load_global_configs()
    
    # Run the base MCP server with OrPaynter extensions
    base_mcp_run(port=port, host=host, addition_custom_mcp_py_file=addition_custom_mcp_py_file)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='OrPaynter Financial MCP Server')
    parser.add_argument('-p', '--port', type=int, default=11999, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('-c', '--custom-mcp', type=str, help='Custom MCP file path')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    try:
        mcp_run(
            port=args.port,
            host=args.host,
            addition_custom_mcp_py_file=args.custom_mcp,
            config_file=args.config
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)
