"""
Main entry point for OrPaynter Financial MCP Server
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from orpaynter_finance_mcp import mcp_run

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OrPaynter Financial MCP Server')
    parser.add_argument('-p', '--port', type=int, default=11999, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('-c', '--custom-mcp', type=str, help='Custom MCP file path')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--version', action='version', version='OrPaynter Finance MCP 1.0.0')
    
    args = parser.parse_args()
    
    try:
        mcp_run(
            port=args.port,
            host=args.host,
            addition_custom_mcp_py_file=args.custom_mcp,
            config_file=args.config
        )
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
