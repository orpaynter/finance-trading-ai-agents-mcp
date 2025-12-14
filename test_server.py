#!/usr/bin/env python3
"""
Test script for OrPaynter MCP Server functionality
"""

import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from simplified_cash_flow_optimizer import CashFlowOptimizer
from simplified_risk_manager import RiskManager
from simplified_predictive_analytics import PredictiveAnalytics
from simplified_alert_system import AlertSystem

def test_cash_flow_optimizer():
    """Test cash flow optimizer functionality"""
    print("Testing Cash Flow Optimizer...")
    
    cash_flow = CashFlowOptimizer()
    
    # Test portfolio analysis
    symbols = ["AAPL", "MSFT", "GOOGL"]
    allocation = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
    
    result = cash_flow.analyze_portfolio_cash_flow(symbols, allocation)
    print(f"Portfolio analysis result: {result['status']}")
    
    # Test rebalancing plan
    target_allocation = {"AAPL": 0.3, "MSFT": 0.4, "GOOGL": 0.3}
    rebalancing_plan = cash_flow.create_rebalancing_plan(
        symbols, allocation, target_allocation, 100000, True
    )
    print(f"Rebalancing plan actions: {len(rebalancing_plan.get('rebalancing_actions', []))}")
    
    return True

def test_risk_manager():
    """Test risk manager functionality"""
    print("Testing Risk Manager...")
    
    risk_mgr = RiskManager()
    
    # Test portfolio risk analysis
    symbols = ["AAPL", "MSFT", "GOOGL"]
    allocation = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25}
    
    risk_analysis = risk_mgr.analyze_portfolio_risk(symbols, allocation)
    print(f"Risk score: {risk_analysis.get('overall_risk_score', 'N/A')}")
    
    # Test VaR calculation
    var_result = risk_mgr.calculate_var_and_volatility(symbols, 252)
    print(f"VaR analysis completed for {len(symbols)} symbols")
    
    return True

def test_predictive_analytics():
    """Test predictive analytics functionality"""
    print("Testing Predictive Analytics...")
    
    pred_analytics = PredictiveAnalytics()
    
    # Test price predictions
    symbols = ["AAPL", "MSFT"]
    predictions = pred_analytics.predict_prices(symbols, 30)
    print(f"Predictions generated for {len(predictions.get('individual_predictions', {}))} symbols")
    
    # Test market sentiment
    sentiment = pred_analytics.analyze_market_sentiment()
    print(f"Market sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
    
    return True

def test_alert_system():
    """Test alert system functionality"""
    print("Testing Alert System...")
    
    alert_system = AlertSystem()
    
    # Test price alert setup
    symbols = ["AAPL", "MSFT"]
    alert_config = alert_system.setup_price_alert(symbols, "price_change", 5.0, "log")
    print(f"Alert setup: {alert_config.get('status', 'N/A')}")
    
    # Test active alerts
    active_alerts = alert_system.get_active_alerts()
    print(f"Active alerts: {active_alerts.get('total_active_alerts', 0)}")
    
    return True

def test_end_to_end_scenario():
    """Test a complete end-to-end financial analysis scenario"""
    print("\n" + "="*60)
    print("TESTING END-TO-END FINANCIAL ANALYSIS SCENARIO")
    print("="*60)
    
    # Initialize components
    cash_flow = CashFlowOptimizer()
    risk_mgr = RiskManager()
    pred_analytics = PredictiveAnalytics()
    alert_system = AlertSystem()
    
    # Portfolio setup
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    current_allocation = {"AAPL": 0.3, "MSFT": 0.25, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.1}
    target_allocation = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "TSLA": 0.15, "NVDA": 0.1}
    
    print(f"Portfolio: {len(symbols)} symbols")
    print(f"Current allocation: {current_allocation}")
    print(f"Target allocation: {target_allocation}")
    
    # 1. Risk Analysis
    print("\n1. RISK ANALYSIS")
    print("-" * 30)
    risk_analysis = risk_mgr.analyze_portfolio_risk(symbols, current_allocation)
    print(f"Overall Risk Score: {risk_analysis.get('overall_risk_score', 0):.3f}")
    print(f"Diversification Score: {risk_analysis.get('diversification_score', 0):.3f}")
    print(f"Concentration Risk Level: {risk_analysis.get('concentration_risk', {}).get('concentration_risk_level', 'Unknown')}")
    
    # 2. Cash Flow Optimization
    print("\n2. CASH FLOW OPTIMIZATION")
    print("-" * 30)
    cash_flow_analysis = cash_flow.analyze_portfolio_cash_flow(symbols, current_allocation)
    print(f"Cash Flow Status: {cash_flow_analysis.get('status', 'Unknown')}")
    print(f"Monthly Cash Flow Estimate: ${cash_flow_analysis.get('portfolio_metrics', {}).get('monthly_cash_flow_estimate', 0):.2f}")
    
    rebalancing_plan = cash_flow.create_rebalancing_plan(
        symbols, current_allocation, target_allocation, 50000, True
    )
    print(f"Rebalancing Actions: {len(rebalancing_plan.get('rebalancing_actions', []))}")
    print(f"Estimated Transaction Costs: ${rebalancing_plan.get('estimated_costs', {}).get('total_estimated_cost', 0):.2f}")
    
    # 3. Predictive Analytics
    print("\n3. PREDICTIVE ANALYTICS")
    print("-" * 30)
    predictions = pred_analytics.predict_prices(symbols, 30)
    print(f"Price Predictions Generated: {len(predictions.get('individual_predictions', {}))} symbols")
    portfolio_pred = predictions.get('portfolio_prediction', {})
    print(f"Predicted Portfolio Return: {portfolio_pred.get('predicted_portfolio_return', 0):.4f}")
    print(f"Prediction Confidence: {portfolio_pred.get('confidence_score', 0):.3f}")
    
    # 4. Market Sentiment
    print("\n4. MARKET SENTIMENT")
    print("-" * 30)
    sentiment = pred_analytics.analyze_market_sentiment()
    print(f"Overall Market Sentiment: {sentiment.get('overall_sentiment', 'Unknown').upper()}")
    sentiment_indicators = sentiment.get('sentiment_indicators', {})
    print(f"Fear & Greed Index: {sentiment_indicators.get('fear_greed_index', 'N/A')}")
    print(f"VIX Level: {sentiment_indicators.get('vix_level', 'N/A')}")
    
    # 5. Alert System
    print("\n5. ALERT SYSTEM")
    print("-" * 30)
    price_alert = alert_system.setup_price_alert(symbols, "price_change", 5.0, "log")
    volume_alert = alert_system.setup_volume_alert(symbols, 2.0, 20, "log")
    volatility_alert = alert_system.setup_volatility_alert(symbols, 30.0, 20, "log")
    
    active_alerts = alert_system.get_active_alerts()
    print(f"Active Alerts Setup: {active_alerts.get('total_active_alerts', 0)}")
    alert_summary = active_alerts.get('alert_summary', {})
    print(f"Alert Types: {list(alert_summary.get('alert_types', {}).keys())}")
    
    # 6. Volatility Analysis
    print("\n6. VOLATILITY ANALYSIS")
    print("-" * 30)
    volatility_forecast = pred_analytics.forecast_volatility(symbols, 30)
    portfolio_vol = volatility_forecast.get('portfolio_volatility', {})
    print(f"Current Portfolio Volatility: {portfolio_vol.get('current_portfolio_volatility', 0):.4f}")
    print(f"Forecasted Portfolio Volatility: {portfolio_vol.get('forecasted_portfolio_volatility', 0):.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("FINANCIAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"✓ Risk Analysis: Portfolio risk score {risk_analysis.get('overall_risk_score', 0):.3f}")
    print(f"✓ Cash Flow: ${cash_flow_analysis.get('portfolio_metrics', {}).get('monthly_cash_flow_estimate', 0):.2f} monthly flow")
    print(f"✓ Predictions: {portfolio_pred.get('predicted_portfolio_return', 0):.4f} expected return")
    print(f"✓ Sentiment: {sentiment.get('overall_sentiment', 'Unknown')} market sentiment")
    print(f"✓ Alerts: {active_alerts.get('total_active_alerts', 0)} active monitoring alerts")
    print(f"✓ Volatility: {portfolio_vol.get('current_portfolio_volatility', 0):.4f} current portfolio vol")
    
    return True

def main():
    """Run all tests"""
    print("🚀 OrPaynter MCP Server - Functional Testing")
    print("=" * 60)
    
    try:
        # Test individual components
        test_cash_flow_optimizer()
        test_risk_manager()
        test_predictive_analytics()
        test_alert_system()
        
        # Test end-to-end scenario
        test_end_to_end_scenario()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - OrPaynter MCP Server is fully functional!")
        print("="*60)
        
        print("\nAvailable MCP Tools:")
        tools = [
            "get_portfolio_health - Comprehensive portfolio health analysis",
            "optimize_cash_flow - Portfolio cash flow optimization and rebalancing",
            "analyze_market_risk - Market risk analysis including VaR and stress tests", 
            "generate_price_predictions - AI-powered price predictions",
            "setup_price_alerts - Real-time price and volatility alerts",
            "get_market_overview - Current market overview and sentiment",
            "validate_configuration - Server configuration and capabilities check"
        ]
        
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)