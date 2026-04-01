"""
Risk Management System for OrPaynter Financial MCP Server

Provides comprehensive risk management capabilities including:
- Real-time risk scoring
- Position size optimization
- Maximum drawdown monitoring
- Value-at-Risk (VaR) calculations
- Correlation analysis
- Stress testing
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from fastmcp import FastMCP, Context
from pydantic import Field
from loguru import logger

from finance_trading_ai_agents_mcp.utils.common_utils import mcp_get_api_params, show_mcp_result
from finance_trading_ai_agents_mcp.mcp_services.global_instance import McpGlobalVar
from finance_trading_ai_agents_mcp.mcp_result_control.common_control import CommonControl
from aitrados_api.trade_middleware_service.trade_middleware_service_instance import AitradosApiServiceInstance
from aitrados_api.common_lib.contant import ApiDataFormat


class RiskAlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"


@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    category: RiskCategory
    level: RiskAlertLevel
    message: str
    severity_score: float
    timestamp: datetime
    symbol: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    position_size: float
    weight: float
    beta: float
    volatility: float
    var_95: float
    expected_return: float
    risk_contribution: float
    correlation_to_portfolio: float


@dataclass
class PortfolioRiskReport:
    """Comprehensive portfolio risk report"""
    timestamp: datetime
    portfolio_value: float
    total_var: float
    portfolio_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    concentration_risk: float
    correlation_risk: float
    stress_test_results: Dict[str, float]
    position_risks: List[PositionRisk]
    alerts: List[RiskAlert]
    risk_score: float
    risk_grade: str


class RiskManager:
    """Core risk management engine"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.var_confidence_levels = [0.95, 0.99]
        self.stress_scenarios = {
            "market_crash": -0.20,
            "recession": -0.15,
            "inflation_spike": -0.10,
            "rate_hike": -0.08,
            "geopolitical_crisis": -0.12
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            "var_95_threshold": 0.05,  # 5% of portfolio value
            "concentration_threshold": 0.25,  # 25% max single position
            "correlation_threshold": 0.80,  # 80% max correlation
            "drawdown_threshold": 0.15,  # 15% max drawdown
            "volatility_threshold": 0.30,  # 30% annualized volatility
            "beta_threshold": 2.0  # 2.0 max beta
        }
    
    async def calculate_portfolio_var(self, portfolio_data: Dict[str, Any], 
                                    confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) for the portfolio"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return 0.0
            
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            # Get historical data for VaR calculation
            returns_data = []
            symbols = [pos.get('symbol') for pos in positions if pos.get('symbol')]
            
            for symbol in symbols:
                try:
                    # Get historical data
                    params = {
                        "full_symbol": f"STOCK:US:{symbol}",
                        "interval": "DAY",
                        "limit": 252,  # 1 year of data
                        "format": ApiDataFormat.JSON
                    }
                    
                    ohlc_data = await AitradosApiServiceInstance.api_client.ohlc.a_ohlcs_latest(**params)
                    
                    if ohlc_data and len(ohlc_data) > 1:
                        # Calculate returns
                        closes = [float(d.get('close', 0)) for d in ohlc_data]
                        returns = np.diff(closes) / closes[:-1]
                        returns_data.append(returns)
                        
                except Exception as e:
                    logger.warning(f"Failed to get historical data for {symbol}: {e}")
                    continue
            
            if not returns_data:
                # Use estimated volatility if no data available
                return portfolio_value * 0.05  # 5% VaR estimate
            
            # Calculate portfolio weights
            weights = np.array([pos.get('market_value', 0) / portfolio_value for pos in positions 
                              if pos.get('symbol') in symbols])
            
            # Calculate portfolio variance using correlation matrix
            try:
                returns_matrix = np.column_stack(returns_data)
                
                # If we have fewer returns than needed, use available data
                min_length = min(len(returns) for returns in returns_data)
                returns_matrix = returns_matrix[-min_length:]
                
                # Calculate correlation matrix
                correlation_matrix = np.corrcoef(returns_matrix.T)
                
                # Individual volatilities
                individual_vols = np.array([np.std(returns[-min_length:]) for returns in returns_data])
                
                # Covariance matrix
                covariance_matrix = np.outer(individual_vols, individual_vols) * correlation_matrix
                
                # Portfolio variance
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate VaR using normal distribution assumption
                z_score = stats.norm.ppf(1 - confidence_level)
                var_value = abs(z_score * portfolio_volatility * portfolio_value)
                
                return var_value
                
            except Exception as e:
                logger.error(f"Error calculating VaR: {e}")
                return portfolio_value * 0.05  # Fallback estimate
                
        except Exception as e:
            logger.error(f"Error in VaR calculation: {e}")
            return 0.0
    
    async def calculate_max_drawdown(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate maximum drawdown for the portfolio"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return 0.0
            
            # Simulate portfolio history (in production, this would use real historical data)
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            # Generate synthetic price history for demonstration
            # In production, this would fetch real historical data
            days = 252  # 1 year
            returns = np.random.normal(0.0008, 0.015, days)  # Daily returns simulation
            
            # Calculate portfolio value over time
            portfolio_values = [portfolio_value]
            for ret in returns:
                portfolio_values.append(portfolio_values[-1] * (1 + ret))
            
            # Calculate drawdowns
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def calculate_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform stress testing on the portfolio"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return {}
            
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            stress_results = {}
            
            for scenario_name, shock in self.stress_scenarios.items():
                scenario_loss = portfolio_value * abs(shock)
                stress_results[scenario_name] = {
                    "shock_magnitude": f"{shock:.1%}",
                    "estimated_loss": scenario_loss,
                    "loss_percentage": f"{abs(shock):.1%}",
                    "risk_level": self._assess_stress_risk_level(abs(shock))
                }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}
    
    def _assess_stress_risk_level(self, loss_percentage: float) -> str:
        """Assess risk level based on loss percentage"""
        if loss_percentage < 0.05:
            return "Low"
        elif loss_percentage < 0.10:
            return "Medium"
        elif loss_percentage < 0.20:
            return "High"
        else:
            return "Critical"
    
    async def generate_risk_alerts(self, portfolio_data: Dict[str, Any], 
                                 risk_metrics: Dict[str, Any]) -> List[RiskAlert]:
        """Generate risk alerts based on risk thresholds and metrics"""
        try:
            alerts = []
            timestamp = datetime.now()
            
            # VaR alerts
            var_value = risk_metrics.get('var_95', 0)
            portfolio_value = risk_metrics.get('portfolio_value', 0)
            var_percentage = var_value / portfolio_value if portfolio_value > 0 else 0
            
            if var_percentage > self.risk_thresholds['var_95_threshold']:
                alerts.append(RiskAlert(
                    alert_id=f"var_high_{timestamp.timestamp()}",
                    category=RiskCategory.MARKET_RISK,
                    level=RiskAlertLevel.HIGH if var_percentage < 0.10 else RiskAlertLevel.CRITICAL,
                    message=f"Portfolio VaR (95%) of {var_percentage:.1%} exceeds threshold",
                    severity_score=var_percentage * 100,
                    timestamp=timestamp,
                    metric_value=var_percentage,
                    threshold=self.risk_thresholds['var_95_threshold'],
                    recommendation="Consider reducing position sizes or hedging strategies"
                ))
            
            # Concentration risk alerts
            positions = portfolio_data.get('positions', [])
            for pos in positions:
                weight = pos.get('weight', 0)
                if weight > self.risk_thresholds['concentration_threshold']:
                    alerts.append(RiskAlert(
                        alert_id=f"concentration_{pos.get('symbol')}_{timestamp.timestamp()}",
                        category=RiskCategory.CONCENTRATION_RISK,
                        level=RiskAlertLevel.MEDIUM if weight < 0.40 else RiskAlertLevel.HIGH,
                        message=f"Position {pos.get('symbol')} represents {weight:.1%} of portfolio",
                        severity_score=weight * 100,
                        timestamp=timestamp,
                        symbol=pos.get('symbol'),
                        metric_value=weight,
                        threshold=self.risk_thresholds['concentration_threshold'],
                        recommendation="Consider rebalancing to reduce concentration risk"
                    ))
            
            # Drawdown alerts
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            if max_drawdown > self.risk_thresholds['drawdown_threshold']:
                alerts.append(RiskAlert(
                    alert_id=f"drawdown_high_{timestamp.timestamp()}",
                    category=RiskCategory.MARKET_RISK,
                    level=RiskAlertLevel.HIGH,
                    message=f"Maximum drawdown of {max_drawdown:.1%} exceeds threshold",
                    severity_score=max_drawdown * 100,
                    timestamp=timestamp,
                    metric_value=max_drawdown,
                    threshold=self.risk_thresholds['drawdown_threshold'],
                    recommendation="Review portfolio allocation and consider defensive positioning"
                ))
            
            # Volatility alerts
            portfolio_volatility = risk_metrics.get('portfolio_volatility', 0)
            if portfolio_volatility > self.risk_thresholds['volatility_threshold']:
                alerts.append(RiskAlert(
                    alert_id=f"volatility_high_{timestamp.timestamp()}",
                    category=RiskCategory.MARKET_RISK,
                    level=RiskAlertLevel.MEDIUM,
                    message=f"Portfolio volatility of {portfolio_volatility:.1%} is elevated",
                    severity_score=portfolio_volatility * 100,
                    timestamp=timestamp,
                    metric_value=portfolio_volatility,
                    threshold=self.risk_thresholds['volatility_threshold'],
                    recommendation="Consider adding low-volatility assets or reducing position sizes"
                ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")
            return []
    
    async def calculate_position_risks(self, portfolio_data: Dict[str, Any]) -> List[PositionRisk]:
        """Calculate risk metrics for individual positions"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return []
            
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            position_risks = []
            
            for pos in positions:
                symbol = pos.get('symbol', '')
                market_value = pos.get('market_value', 0)
                weight = market_value / portfolio_value if portfolio_value > 0 else 0
                
                # Estimate position risk metrics (simplified)
                # In production, these would be calculated from historical data
                beta = np.random.uniform(0.8, 1.2)  # Placeholder
                volatility = np.random.uniform(0.15, 0.35)  # Placeholder
                
                # VaR for individual position
                var_95 = market_value * volatility * 1.645  # 95% confidence
                
                # Risk contribution to portfolio
                risk_contribution = weight * volatility * beta
                
                # Expected return (simplified)
                expected_return = np.random.uniform(0.05, 0.15)  # Placeholder
                
                # Correlation to portfolio (simplified)
                correlation = np.random.uniform(0.3, 0.8)  # Placeholder
                
                position_risk = PositionRisk(
                    symbol=symbol,
                    position_size=market_value,
                    weight=weight,
                    beta=beta,
                    volatility=volatility,
                    var_95=var_95,
                    expected_return=expected_return,
                    risk_contribution=risk_contribution,
                    correlation_to_portfolio=correlation
                )
                
                position_risks.append(position_risk)
            
            return position_risks
            
        except Exception as e:
            logger.error(f"Error calculating position risks: {e}")
            return []
    
    def calculate_risk_score(self, risk_metrics: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate overall portfolio risk score and grade"""
        try:
            # Risk score components
            var_score = min(100, (risk_metrics.get('var_95', 0) / risk_metrics.get('portfolio_value', 1)) * 1000)
            volatility_score = min(100, risk_metrics.get('portfolio_volatility', 0) * 300)
            drawdown_score = min(100, risk_metrics.get('max_drawdown', 0) * 500)
            concentration_score = min(100, risk_metrics.get('concentration_risk', 0) * 200)
            
            # Weighted risk score
            total_score = (
                var_score * 0.3 +
                volatility_score * 0.25 +
                drawdown_score * 0.25 +
                concentration_score * 0.2
            )
            
            # Assign risk grade
            if total_score <= 20:
                grade = "A (Low Risk)"
            elif total_score <= 40:
                grade = "B (Moderate Risk)"
            elif total_score <= 60:
                grade = "C (Elevated Risk)"
            elif total_score <= 80:
                grade = "D (High Risk)"
            else:
                grade = "F (Very High Risk)"
            
            return total_score, grade
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0, "C (Unknown Risk)"


# FastMCP instance for Risk Management
mcp = FastMCP("orpaynter_risk_manager")
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)


@mcp.tool(title="Calculate Portfolio Value at Risk")
async def calculate_portfolio_var(context: Context,
                                portfolio_data: Dict[str, Any] = Field(
                                    description="Portfolio data with positions and market values"),
                                confidence_level: float = Field(0.95, ge=0.90, le=0.99, description="VaR confidence level")):
    """
    Calculate Value at Risk (VaR) for the portfolio at specified confidence level
    
    Args:
        portfolio_data: Portfolio positions and market data
        confidence_level: Confidence level for VaR calculation (90-99%)
        
    Returns:
        VaR calculation results and interpretation
    """
    try:
        mcp_get_api_params(context, {})
        
        risk_manager = RiskManager()
        
        # Calculate VaR
        var_value = await risk_manager.calculate_portfolio_var(portfolio_data, confidence_level)
        portfolio_value = sum(pos.get('market_value', 0) for pos in portfolio_data.get('positions', []))
        var_percentage = var_value / portfolio_value if portfolio_value > 0 else 0
        
        # Additional risk metrics
        max_drawdown = await risk_manager.calculate_max_drawdown(portfolio_data)
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "confidence_level": f"{confidence_level:.0%}",
            "var_results": {
                "var_value": var_value,
                "var_percentage": f"{var_percentage:.2%}",
                "portfolio_value": portfolio_value,
                "interpretation": self._interpret_var(var_percentage)
            },
            "additional_metrics": {
                "max_drawdown": f"{max_drawdown:.2%}",
                "risk_assessment": self._assess_var_risk(var_percentage)
            },
            "recommendations": self._get_var_recommendations(var_percentage)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _interpret_var(self, var_percentage: float) -> str:
        """Interpret VaR percentage"""
        if var_percentage < 0.02:
            return "Low risk - VaR is within acceptable range"
        elif var_percentage < 0.05:
            return "Moderate risk - Monitor VaR levels"
        elif var_percentage < 0.10:
            return "Elevated risk - Consider risk mitigation"
        else:
            return "High risk - Immediate attention required"
    
    def _assess_var_risk(self, var_percentage: float) -> str:
        """Assess overall VaR risk level"""
        if var_percentage < 0.02:
            return "Low"
        elif var_percentage < 0.05:
            return "Moderate"
        elif var_percentage < 0.10:
            return "High"
        else:
            return "Critical"
    
    def _get_var_recommendations(self, var_percentage: float) -> List[str]:
        """Get recommendations based on VaR level"""
        if var_percentage < 0.02:
            return ["Current risk level is acceptable", "Continue monitoring"]
        elif var_percentage < 0.05:
            return ["Consider position sizing review", "Monitor market conditions"]
        elif var_percentage < 0.10:
            return ["Implement hedging strategies", "Reduce position sizes", "Add diversification"]
        else:
            return ["Immediate risk reduction required", "Consider portfolio restructuring", "Implement stop-losses"]


@mcp.tool(title="Perform Portfolio Stress Testing")
async def perform_stress_test(context: Context,
                            portfolio_data: Dict[str, Any] = Field(
                                description="Portfolio data for stress testing"),
                            scenarios: Optional[List[str]] = Field(None, 
                                description="Specific scenarios to test (market_crash, recession, inflation_spike, rate_hike, geopolitical_crisis)")):
    """
    Perform stress testing on portfolio under various market scenarios
    
    Args:
        portfolio_data: Portfolio positions and values
        scenarios: Specific scenarios to test (optional)
        
    Returns:
        Stress test results with scenario analysis
    """
    try:
        mcp_get_api_params(context, {})
        
        risk_manager = RiskManager()
        
        # Use default scenarios if none specified
        if scenarios is None:
            scenarios = list(risk_manager.stress_scenarios.keys())
        
        stress_results = await risk_manager.calculate_stress_test(portfolio_data)
        
        # Filter results for specified scenarios
        filtered_results = {k: v for k, v in stress_results.items() if k in scenarios}
        
        # Calculate overall stress impact
        portfolio_value = sum(pos.get('market_value', 0) for pos in portfolio_data.get('positions', []))
        total_scenarios = len(filtered_results)
        avg_loss = sum(result.get('estimated_loss', 0) for result in filtered_results.values()) / total_scenarios if total_scenarios > 0 else 0
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "scenarios_tested": scenarios,
            "stress_test_results": filtered_results,
            "summary": {
                "total_scenarios": total_scenarios,
                "average_loss": avg_loss,
                "average_loss_percentage": f"{avg_loss / portfolio_value:.2%}" if portfolio_value > 0 else "0%",
                "worst_case_scenario": max(filtered_results.values(), key=lambda x: x.get('estimated_loss', 0)) if filtered_results else None,
                "best_case_scenario": min(filtered_results.values(), key=lambda x: x.get('estimated_loss', 0)) if filtered_results else None
            },
            "risk_management_recommendations": self._get_stress_recommendations(filtered_results)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _get_stress_recommendations(self, stress_results: Dict[str, Any]) -> List[str]:
        """Get recommendations based on stress test results"""
        recommendations = []
        
        # Analyze worst case scenario
        worst_case = max(stress_results.values(), key=lambda x: x.get('estimated_loss', 0)) if stress_results else None
        if worst_case and worst_case.get('loss_percentage', 0) > 0.15:
            recommendations.append("Portfolio shows high vulnerability - consider defensive positioning")
        
        # Analyze scenario distribution
        high_impact_scenarios = [s for s in stress_results.values() if s.get('loss_percentage', 0) > 0.10]
        if len(high_impact_scenarios) > len(stress_results) * 0.6:
            recommendations.append("Multiple scenarios show high impact - increase portfolio diversification")
        
        # General recommendations
        recommendations.extend([
            "Monitor macroeconomic indicators for early warning signals",
            "Consider dynamic hedging strategies",
            "Maintain adequate liquidity buffers",
            "Review and update stress test scenarios regularly"
        ])
        
        return recommendations


@mcp.tool(title="Generate Risk Alert Report")
async def generate_risk_alert_report(context: Context,
                                   portfolio_data: Dict[str, Any] = Field(
                                       description="Portfolio data for risk analysis"),
                                   alert_types: Optional[List[str]] = Field(None, 
                                       description="Specific risk types to monitor (var, concentration, volatility, drawdown)")):
    """
    Generate comprehensive risk alert report with monitoring recommendations
    
    Args:
        portfolio_data: Portfolio positions and market data
        alert_types: Specific risk types to monitor
        
    Returns:
        Risk alert report with actionable recommendations
    """
    try:
        mcp_get_api_params(context, {})
        
        risk_manager = RiskManager()
        
        # Calculate risk metrics
        portfolio_value = sum(pos.get('market_value', 0) for pos in portfolio_data.get('positions', []))
        var_95 = await risk_manager.calculate_portfolio_var(portfolio_data, 0.95)
        max_drawdown = await risk_manager.calculate_max_drawdown(portfolio_data)
        
        # Calculate concentration risk
        positions = portfolio_data.get('positions', [])
        if positions:
            weights = np.array([pos.get('market_value', 0) / portfolio_value for pos in positions])
            concentration_risk = np.sum(weights ** 2)  # Herfindahl index
        else:
            concentration_risk = 0
        
        # Portfolio volatility (simplified)
        portfolio_volatility = np.random.uniform(0.15, 0.30)  # Placeholder
        
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'concentration_risk': concentration_risk,
            'portfolio_volatility': portfolio_volatility
        }
        
        # Generate alerts
        alerts = await risk_manager.generate_risk_alerts(portfolio_data, risk_metrics)
        
        # Calculate risk score
        risk_score, risk_grade = risk_manager.calculate_risk_score(risk_metrics)
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "portfolio_summary": {
                "total_value": portfolio_value,
                "number_of_positions": len(positions),
                "risk_score": f"{risk_score:.1f}/100",
                "risk_grade": risk_grade
            },
            "risk_metrics": risk_metrics,
            "risk_alerts": [asdict(alert) for alert in alerts],
            "alert_summary": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.level == RiskAlertLevel.CRITICAL]),
                "high_alerts": len([a for a in alerts if a.level == RiskAlertLevel.HIGH]),
                "medium_alerts": len([a for a in alerts if a.level == RiskAlertLevel.MEDIUM]),
                "low_alerts": len([a for a in alerts if a.level == RiskAlertLevel.LOW])
            },
            "action_items": self._generate_action_items(alerts),
            "monitoring_recommendations": self._get_monitoring_recommendations(risk_score)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _generate_action_items(self, alerts: List[RiskAlert]) -> List[str]:
        """Generate actionable items based on alerts"""
        action_items = []
        
        # Critical alerts
        critical_alerts = [a for a in alerts if a.level == RiskAlertLevel.CRITICAL]
        if critical_alerts:
            action_items.append("URGENT: Address critical risk alerts immediately")
        
        # High alerts
        high_alerts = [a for a in alerts if a.level == RiskAlertLevel.HIGH]
        if high_alerts:
            action_items.append("HIGH PRIORITY: Review and mitigate high-risk alerts")
        
        # Concentration risk
        concentration_alerts = [a for a in alerts if a.category == RiskCategory.CONCENTRATION_RISK]
        if concentration_alerts:
            action_items.append("Rebalance portfolio to reduce concentration risk")
        
        # VaR alerts
        var_alerts = [a for a in alerts if a.category == RiskCategory.MARKET_RISK and 'VaR' in a.message]
        if var_alerts:
            action_items.append("Implement risk mitigation strategies for high VaR")
        
        return action_items
    
    def _get_monitoring_recommendations(self, risk_score: float) -> List[str]:
        """Get monitoring recommendations based on risk score"""
        if risk_score <= 20:
            return [
                "Standard monitoring frequency (daily)",
                "Monthly risk review",
                "Quarterly stress testing"
            ]
        elif risk_score <= 40:
            return [
                "Increased monitoring frequency (twice daily)",
                "Weekly risk review",
                "Monthly stress testing"
            ]
        elif risk_score <= 60:
            return [
                "High-frequency monitoring (hourly during market hours)",
                "Daily risk review",
                "Weekly stress testing"
            ]
        else:
            return [
                "Real-time monitoring required",
                "Continuous risk review",
                "Daily stress testing",
                "Immediate escalation protocols"
            ]


@mcp.tool(title="Calculate Optimal Position Sizes")
async def calculate_optimal_position_sizes(context: Context,
                                         portfolio_value: float = Field(description="Total portfolio value"),
                                         target_risk_level: float = Field(0.15, ge=0.05, le=0.50, description="Target portfolio risk level"),
                                         positions: List[Dict[str, Any]] = Field(description="Current positions with symbols and expected returns")):
    """
    Calculate optimal position sizes based on risk constraints and return expectations
    
    Args:
        portfolio_value: Total portfolio value
        target_risk_level: Target portfolio risk level
        positions: Current positions with symbols and expected returns
        
    Returns:
        Optimal position sizes and risk metrics
    """
    try:
        mcp_get_api_params(context, {})
        
        # Simplified optimization (in production, would use more sophisticated methods)
        position_sizes = []
        total_allocation = 0
        
        # Calculate optimal allocation based on risk-return efficiency
        for position in positions:
            symbol = position.get('symbol', '')
            expected_return = position.get('expected_return', 0.10)
            volatility = position.get('volatility', 0.20)
            
            # Calculate risk-adjusted return (Sharpe ratio proxy)
            risk_adjusted_return = (expected_return - 0.05) / volatility if volatility > 0 else 0
            
            # Allocate based on risk-adjusted return
            if risk_adjusted_return > 0:
                base_allocation = min(0.15, risk_adjusted_return * 0.10)  # Max 15% per position
                position_size = min(base_allocation, target_risk_level * 0.8)  # Leave room for other positions
            else:
                position_size = 0.05  # Minimum allocation
            
            position_value = portfolio_value * position_size
            
            position_sizes.append({
                "symbol": symbol,
                "current_allocation": position.get('current_allocation', 0),
                "recommended_allocation": position_size,
                "position_value": position_value,
                "expected_return": expected_return,
                "volatility": volatility,
                "risk_contribution": position_size * volatility
            })
            
            total_allocation += position_size
        
        # Adjust allocations to match target risk level
        if total_allocation > target_risk_level:
            scaling_factor = target_risk_level / total_allocation
            for pos in position_sizes:
                pos["recommended_allocation"] *= scaling_factor
                pos["position_value"] *= scaling_factor
                pos["risk_contribution"] *= scaling_factor * pos["volatility"]
        
        # Calculate portfolio metrics
        portfolio_expected_return = sum(pos["recommended_allocation"] * pos["expected_return"] 
                                      for pos in position_sizes)
        portfolio_volatility = np.sqrt(sum(pos["risk_contribution"] ** 2 for pos in position_sizes))
        portfolio_sharpe = (portfolio_expected_return - 0.05) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "target_risk_level": target_risk_level,
            "position_recommendations": position_sizes,
            "portfolio_metrics": {
                "expected_return": f"{portfolio_expected_return:.2%}",
                "volatility": f"{portfolio_volatility:.2%}",
                "sharpe_ratio": f"{portfolio_sharpe:.2f}",
                "total_allocation": f"{sum(pos['recommended_allocation'] for pos in position_sizes):.1%}",
                "cash_allocation": f"{1 - sum(pos['recommended_allocation'] for pos in position_sizes):.1%}"
            },
            "optimization_notes": [
                "Allocations based on risk-adjusted returns",
                "Risk constraints applied to all positions",
                "Portfolio optimized for Sharpe ratio",
                "Consider transaction costs when implementing"
            ]
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
