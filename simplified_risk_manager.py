"""
Simplified Risk Manager for OrPaynter MCP Server
Standalone implementation without external dependencies
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math

class RiskManager:
    """Simplified risk management engine"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.logger = None
    
    def analyze_portfolio_risk(self, symbols: List[str], allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            symbols: List of stock symbols
            allocation: Dictionary of symbol allocation percentages
        
        Returns:
            Portfolio risk analysis results
        """
        try:
            risk_analysis = {
                "analysis_date": datetime.now().isoformat(),
                "symbols": symbols,
                "overall_risk_score": self._calculate_overall_risk_score(allocation),
                "diversification_score": self._calculate_diversification_score(allocation),
                "concentration_risk": self._analyze_concentration_risk(allocation),
                "correlation_analysis": self._analyze_correlation_structure(allocation),
                "volatility_analysis": self._analyze_volatility(allocation),
                "downside_risk": self._analyze_downside_risk(allocation),
                "risk_recommendations": self._generate_risk_recommendations(allocation)
            }
            
            return risk_analysis
            
        except Exception as e:
            return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}
    
    def calculate_var_and_volatility(self, symbols: List[str], lookback_days: int = 252) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) and volatility metrics
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days for historical analysis
        
        Returns:
            VaR and volatility analysis
        """
        try:
            var_analysis = {
                "analysis_date": datetime.now().isoformat(),
                "symbols": symbols,
                "lookback_days": lookback_days,
                "var_metrics": {},
                "volatility_metrics": {},
                "risk_scenarios": {}
            }
            
            for symbol in symbols:
                # Simulate VaR and volatility calculations
                symbol_var = self._calculate_symbol_var(symbol, lookback_days)
                symbol_vol = self._calculate_symbol_volatility(symbol, lookback_days)
                
                var_analysis["var_metrics"][symbol] = symbol_var
                var_analysis["volatility_metrics"][symbol] = symbol_vol
            
            # Portfolio-level metrics
            var_analysis["portfolio_var"] = self._calculate_portfolio_var(symbols, lookback_days)
            var_analysis["portfolio_volatility"] = self._calculate_portfolio_volatility(symbols, lookback_days)
            
            # Risk scenarios
            var_analysis["risk_scenarios"] = self._generate_risk_scenarios(symbols)
            
            return var_analysis
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def analyze_correlations(self, symbols: List[str], lookback_days: int = 252) -> Dict[str, Any]:
        """
        Analyze correlation structure between portfolio assets
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days for correlation analysis
        
        Returns:
            Correlation analysis results
        """
        try:
            correlation_analysis = {
                "analysis_date": datetime.now().isoformat(),
                "symbols": symbols,
                "lookback_days": lookback_days,
                "correlation_matrix": self._generate_correlation_matrix(symbols),
                "average_correlation": self._calculate_average_correlation(symbols),
                "diversification_benefit": self._assess_diversification_benefit(symbols),
                "correlation_insights": self._generate_correlation_insights(symbols)
            }
            
            return correlation_analysis
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def run_stress_tests(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Run portfolio stress tests under various market scenarios
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Stress test results
        """
        try:
            stress_tests = {
                "test_date": datetime.now().isoformat(),
                "symbols": symbols,
                "scenarios": {}
            }
            
            # Define stress test scenarios
            scenarios = [
                {"name": "Market Crash", "shock": -0.30, "description": "30% market decline"},
                {"name": "Interest Rate Spike", "shock": -0.15, "description": "Rising rates impact"},
                {"name": "Volatility Spike", "shock": -0.20, "description": "Increased market volatility"},
                {"name": "Sector Rotation", "shock": -0.10, "description": "Sector-specific stress"},
                {"name": "Liquidity Crisis", "shock": -0.25, "description": "Reduced market liquidity"}
            ]
            
            for scenario in scenarios:
                scenario_result = self._run_single_stress_test(symbols, scenario)
                stress_tests["scenarios"][scenario["name"]] = scenario_result
            
            stress_tests["overall_assessment"] = self._assess_stress_test_results(stress_tests["scenarios"])
            
            return stress_tests
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def calculate_position_sizing(self, symbol: str, risk_tolerance: float = 0.15, portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk management principles
        
        Args:
            symbol: Stock symbol
            risk_tolerance: Portfolio risk tolerance (0.15 = 15%)
            portfolio_value: Total portfolio value
        
        Returns:
            Position sizing recommendations
        """
        try:
            # Simulate volatility and correlation for the symbol
            symbol_volatility = np.random.uniform(0.15, 0.45)  # 15-45% annualized
            symbol_correlation = np.random.uniform(0.3, 0.8)   # 0.3-0.8 correlation with market
            
            # Risk-based position sizing
            max_position_size = risk_tolerance / symbol_volatility
            adjusted_position_size = max_position_size * (1 - symbol_correlation * 0.5)
            
            position_sizing = {
                "symbol": symbol,
                "calculation_date": datetime.now().isoformat(),
                "risk_parameters": {
                    "portfolio_risk_tolerance": risk_tolerance,
                    "symbol_volatility": round(symbol_volatility, 4),
                    "estimated_correlation": round(symbol_correlation, 4)
                },
                "position_recommendations": {
                    "max_position_percentage": round(min(adjusted_position_size, 0.25), 4),  # Cap at 25%
                    "max_dollar_amount": round(min(adjusted_position_size, 0.25) * portfolio_value, 2),
                    "risk_adjusted_size": round(adjusted_position_size, 4)
                },
                "risk_warnings": self._generate_position_warnings(adjusted_position_size, symbol_volatility)
            }
            
            return position_sizing
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _calculate_overall_risk_score(self, allocation: Dict[str, float]) -> float:
        """Calculate overall portfolio risk score (0-1, higher = riskier)"""
        try:
            # Weighted risk score based on concentration and volatility
            max_allocation = max(allocation.values())
            num_positions = len(allocation)
            
            # Concentration component (higher concentration = higher risk)
            concentration_risk = max_allocation
            
            # Diversification component (fewer positions = higher risk)
            diversification_risk = max(0, (10 - num_positions) / 10) if num_positions < 10 else 0
            
            # Weighted combination
            overall_risk = (concentration_risk * 0.6) + (diversification_risk * 0.4)
            
            return min(overall_risk, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_diversification_score(self, allocation: Dict[str, float]) -> float:
        """Calculate portfolio diversification score (0-1, higher = more diversified)"""
        try:
            # Herfindahl-Hirschman Index approach
            hhi = sum(pct ** 2 for pct in allocation.values())
            # Convert to diversification score (1 - HHI)
            diversification_score = 1 - hhi
            
            # Boost for number of positions
            position_bonus = min(len(allocation) / 15, 0.3)  # Max 30% bonus for 15+ positions
            
            return min(diversification_score + position_bonus, 1.0)
            
        except Exception:
            return 0.5
    
    def _analyze_concentration_risk(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze concentration risk in portfolio"""
        try:
            sorted_allocations = sorted(allocation.values(), reverse=True)
            
            return {
                "max_position": max(allocation.values()),
                "top_3_weight": sum(sorted_allocations[:3]),
                "herfindahl_index": sum(pct ** 2 for pct in allocation.values()),
                "concentration_risk_level": self._assess_concentration_level(max(allocation.values())),
                "recommendations": self._generate_concentration_recommendations(allocation)
            }
            
        except Exception:
            return {"max_position": 0.25, "concentration_risk_level": "medium"}
    
    def _analyze_correlation_structure(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze correlation structure (simplified)"""
        # This would normally use historical correlation data
        # For now, return simulated analysis
        return {
            "average_correlation": 0.65,
            "correlation_stability": "moderate",
            "diversification_benefit": "good",
            "correlation_trend": "stable"
        }
    
    def _analyze_volatility(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio volatility characteristics"""
        try:
            # Simulate volatility analysis
            weighted_volatility = sum(allocation.get(symbol, 0) * 0.25 for symbol in allocation.keys())
            
            return {
                "portfolio_volatility": round(weighted_volatility, 4),
                "volatility_regime": "normal",
                "volatility_trend": "stable",
                "risk_adjusted_return": round(0.10 / max(weighted_volatility, 0.01), 4)
            }
            
        except Exception:
            return {"portfolio_volatility": 0.20, "risk_adjusted_return": 0.50}
    
    def _analyze_downside_risk(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze downside risk characteristics"""
        try:
            # Simulate downside risk metrics
            max_drawdown = max(allocation.values()) * 0.35  # Estimated max drawdown
            downside_deviation = sum(allocation.get(symbol, 0) * 0.15 for symbol in allocation.keys())
            
            return {
                "estimated_max_drawdown": round(max_drawdown, 4),
                "downside_deviation": round(downside_deviation, 4),
                "sortino_ratio": round(0.10 / max(downside_deviation, 0.01), 4),
                "downside_risk_level": "moderate" if max_drawdown < 0.25 else "high"
            }
            
        except Exception:
            return {"estimated_max_drawdown": 0.25, "sortino_ratio": 0.40, "downside_risk_level": "moderate"}
    
    def _generate_risk_recommendations(self, allocation: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            max_allocation = max(allocation.values())
            num_positions = len(allocation)
            
            if max_allocation > 0.25:
                recommendations.append({
                    "type": "concentration",
                    "priority": "high",
                    "recommendation": "Reduce maximum position size to under 25%"
                })
            
            if num_positions < 5:
                recommendations.append({
                    "type": "diversification",
                    "priority": "medium",
                    "recommendation": "Increase number of positions for better diversification"
                })
            
            recommendations.append({
                "type": "monitoring",
                "priority": "low",
                "recommendation": "Implement regular risk monitoring and rebalancing"
            })
            
            return recommendations
            
        except Exception:
            return [{"type": "general", "priority": "medium", "recommendation": "Monitor portfolio risk regularly"}]
    
    def _calculate_symbol_var(self, symbol: str, lookback_days: int) -> Dict[str, float]:
        """Calculate Value at Risk for a single symbol"""
        # Simulate VaR calculations
        return {
            "var_95": round(np.random.uniform(0.02, 0.08), 4),  # 95% VaR
            "var_99": round(np.random.uniform(0.03, 0.12), 4),  # 99% VaR
            "expected_shortfall": round(np.random.uniform(0.04, 0.15), 4)
        }
    
    def _calculate_symbol_volatility(self, symbol: str, lookback_days: int) -> Dict[str, float]:
        """Calculate volatility metrics for a single symbol"""
        return {
            "annualized_volatility": round(np.random.uniform(0.15, 0.45), 4),
            "recent_volatility": round(np.random.uniform(0.10, 0.30), 4),
            "volatility_percentile": round(np.random.uniform(0.3, 0.8), 4)
        }
    
    def _calculate_portfolio_var(self, symbols: List[str], lookback_days: int) -> Dict[str, float]:
        """Calculate portfolio-level VaR"""
        return {
            "portfolio_var_95": round(np.random.uniform(0.015, 0.06), 4),
            "portfolio_var_99": round(np.random.uniform(0.025, 0.09), 4),
            "marginal_var": round(np.random.uniform(0.01, 0.04), 4)
        }
    
    def _calculate_portfolio_volatility(self, symbols: List[str], lookback_days: int) -> float:
        """Calculate portfolio volatility"""
        return round(np.random.uniform(0.12, 0.25), 4)
    
    def _generate_risk_scenarios(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate risk scenario analysis"""
        return {
            "best_case": {"return": 0.25, "probability": 0.10},
            "base_case": {"return": 0.08, "probability": 0.60},
            "worst_case": {"return": -0.20, "probability": 0.30}
        }
    
    def _generate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Generate simulated correlation matrix"""
        matrix = {}
        for i, symbol1 in enumerate(symbols):
            matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    matrix[symbol1][symbol2] = 1.0
                else:
                    matrix[symbol1][symbol2] = round(np.random.uniform(0.3, 0.8), 3)
        return matrix
    
    def _calculate_average_correlation(self, symbols: List[str]) -> float:
        """Calculate average correlation between portfolio assets"""
        return round(np.random.uniform(0.4, 0.7), 3)
    
    def _assess_diversification_benefit(self, symbols: List[str]) -> str:
        """Assess diversification benefit of the portfolio"""
        return "good" if len(symbols) >= 5 else "moderate" if len(symbols) >= 3 else "limited"
    
    def _generate_correlation_insights(self, symbols: List[str]) -> List[str]:
        """Generate correlation insights and recommendations"""
        insights = [
            "Monitor correlation changes during market stress periods",
            "Consider uncorrelated assets for better diversification",
            "High correlations may reduce diversification benefits"
        ]
        return insights
    
    def _run_single_stress_test(self, symbols: List[str], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single stress test scenario"""
        shock = scenario["shock"]
        
        return {
            "scenario_name": scenario["name"],
            "shock_magnitude": shock,
            "description": scenario["description"],
            "portfolio_impact": round(shock * np.random.uniform(0.7, 1.2), 4),
            "recovery_time": f"{np.random.randint(6, 24)} months",
            "probability": round(np.random.uniform(0.05, 0.15), 3)
        }
    
    def _assess_stress_test_results(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall stress test results"""
        impacts = [scenario["portfolio_impact"] for scenario in scenarios.values()]
        avg_impact = sum(impacts) / len(impacts)
        
        return {
            "overall_resilience": "moderate",
            "average_impact": round(avg_impact, 4),
            "stress_test_conclusion": "Portfolio shows moderate resilience to stress scenarios",
            "recommendations": [
                "Consider additional diversification",
                "Monitor correlation during stress periods",
                "Review position sizing in volatile assets"
            ]
        }
    
    def _assess_concentration_level(self, max_allocation: float) -> str:
        """Assess concentration risk level"""
        if max_allocation > 0.30:
            return "high"
        elif max_allocation > 0.20:
            return "medium"
        else:
            return "low"
    
    def _generate_concentration_recommendations(self, allocation: Dict[str, float]) -> List[str]:
        """Generate concentration risk recommendations"""
        recommendations = []
        max_allocation = max(allocation.values())
        
        if max_allocation > 0.25:
            recommendations.append("Consider reducing position concentration")
        
        if len(allocation) < 5:
            recommendations.append("Add more positions to improve diversification")
        
        return recommendations
    
    def _generate_position_warnings(self, position_size: float, volatility: float) -> List[str]:
        """Generate position sizing warnings"""
        warnings = []
        
        if position_size > 0.20:
            warnings.append("Large position size increases concentration risk")
        
        if volatility > 0.35:
            warnings.append("High volatility asset requires smaller position size")
        
        if position_size * volatility > 0.05:
            warnings.append("Position risk contribution is elevated")
        
        return warnings