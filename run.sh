#!/bin/bash

# OrPaynter Financial MCP Server Startup Script

echo "Starting OrPaynter Financial MCP Server..."

# Set Python path
export PYTHONPATH="/workspace/orpaynter-finance-mcp:$PYTHONPATH"

# Run the standalone server
python3 /workspace/orpaynter-finance-mcp/standalone_server.py
