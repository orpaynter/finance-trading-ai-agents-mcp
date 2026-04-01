"""
Simplified Cash Flow Optimizer for OrPaynter MCP Server
Standalone implementation without external dependencies
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math

class CashFlowOptimizer:
    """Simplified cash flow optimization engine"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.logger = None
    
    def analyze_portfolio_cash_flow(self, symbols: List[str], allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze portfolio cash flow characteristics
        
        Args:
            symbols: List of stock symbols
            allocation: Dictionary of symbol allocation percentages
        
        Returns:
            Cash flow analysis results
        """
        try:
            total_allocation = sum(allocation.values())
            if abs(total_allocation - 1.0) > 0.01:
                return {"status": "invalid_allocation", "message": "Allocation must sum to 100%"}
            
            # Simulate cash flow analysis
            cash_flow_analysis = {
                "status": "analyzed",
                "analysis_date": datetime.now().isoformat(),
                "portfolio_metrics": {
                    "total_symbols": len(symbols),
                    "diversification_score": self._calculate_diversification_score(allocation),
                    "risk_adjusted_return": self._simulate_risk_adjusted_return(allocation),
                    "monthly_cash_flow_estimate": self._estimate_monthly_cash_flow(symbols, allocation)
                },
                "allocation_analysis": {
                    "concentration_risk": self._analyze_concentration(allocation),
                    "sector_balance": self._analyze_sector_balance(symbols)
                },
                "recommendations": self._generate_cash_flow_recommendations(allocation)
            }
            
            return cash_flow_analysis
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def create_rebalancing_plan(
        self, 
        symbols: List[str], 
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        budget: float,
        tax_loss_harvest: bool = True
    ) -> Dict[str, Any]:
        """
        Create portfolio rebalancing plan
        
        Args:
            symbols: List of stock symbols
            current_allocation: Current allocation percentages
            target_allocation: Target allocation percentages
            budget: Available budget for rebalancing
            tax_loss_harvest: Enable tax loss harvesting
        
        Returns:
            Detailed rebalancing plan
        """
        try:
            rebalancing_plan = {
                "plan_date": datetime.now().isoformat(),
                "parameters": {
                    "symbols": symbols,
                    "budget": budget,
                    "tax_loss_harvesting": tax_loss_harvest
                },
                "rebalancing_actions": [],
                "estimated_costs": {},
                "tax_considerations": {},
                "implementation_notes": []
            }
            
            # Calculate allocation differences
            for symbol in symbols:
                current_pct = current_allocation.get(symbol, 0.0)
                target_pct = target_allocation.get(symbol, 0.0)
                difference = target_pct - current_pct
                
                if abs(difference) > 0.01:  # 1% threshold
                    action_type = "buy" if difference > 0 else "sell"
                    amount = abs(difference) * budget
                    
                    rebalancing_plan["rebalancing_actions"].append({
                        "symbol": symbol,
                        "action": action_type,
                        "current_allocation": current_pct,
                        "target_allocation": target_pct,
                        "allocation_change": difference,
                        "estimated_amount": amount,
                        "percentage_change": abs(difference) * 100
                    })
            
            # Calculate estimated costs
            estimated_transaction_costs = len(rebalancing_plan["rebalancing_actions"]) * 9.95  # Typical commission
            rebalancing_plan["estimated_costs"] = {
                "transaction_costs": estimated_transaction_costs,
                "slippage_estimate": budget * 0.001,  # 0.1% slippage
                "total_estimated_cost": estimated_transaction_costs + (budget * 0.001)
            }
            
            # Tax considerations
            if tax_loss_harvest:
                rebalancing_plan["tax_considerations"] = {
                    "tax_loss_harvesting_available": True,
                    "estimated_tax_savings": budget * 0.20 * 0.25,  # Simplified calculation
                    "wash_sale_considerations": "Monitor for 30-day wash sale rule compliance"
                }
            
            rebalancing_plan["implementation_notes"] = [
                "Consider executing trades during high liquidity periods",
                "Monitor market impact for large orders",
                "Review tax implications before execution",
                "Ensure compliance with internal trading policies"
            ]
            
            return rebalancing_plan
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def generate_rebalancing_recommendations(
        self, 
        symbols: List[str], 
        allocation: Dict[str, float],
        risk_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate rebalancing recommendations based on current allocation and risk analysis
        
        Args:
            symbols: List of stock symbols
            allocation: Current allocation dictionary
            risk_analysis: Risk analysis results
        
        Returns:
            List of rebalancing recommendations
        """
        recommendations = []
        
        try:
            # Concentration risk recommendations
            concentration_risk = risk_analysis.get("concentration_risk", {})
            if concentration_risk.get("max_allocation", 0) > 0.25:  # 25% threshold
                recommendations.append({
                    "type": "concentration_risk",
                    "priority": "high",
                    "title": "Reduce Concentration Risk",
                    "description": "Portfolio has high concentration in single positions",
                    "action": "Rebalance to reduce max position size to under 25%"
                })
            
            # Diversification recommendations
            diversification_score = risk_analysis.get("diversification_score", 0)
            if diversification_score < 0.7:  # 70% threshold
                recommendations.append({
                    "type": "diversification",
                    "priority": "medium",
                    "title": "Improve Diversification",
                    "description": "Portfolio could benefit from additional diversification",
                    "action": "Consider adding positions in underrepresented sectors or asset classes"
                })
            
            # Cash flow recommendations
            cash_flow_status = risk_analysis.get("cash_flow_status", "unknown")
            if cash_flow_status in ["negative", "declining"]:
                recommendations.append({
                    "type": "cash_flow",
                    "priority": "high",
                    "title": "Optimize Cash Flow",
                    "description": "Portfolio cash flow is suboptimal",
                    "action": "Consider rebalancing to improve income-generating positions"
                })
            
            # Risk-adjusted return recommendations
            overall_risk_score = risk_analysis.get("overall_risk_score", 0)
            if overall_risk_score > 0.8:  # High risk threshold
                recommendations.append({
                    "type": "risk_management",
                    "priority": "high",
                    "title": "Risk Management",
                    "description": "Portfolio risk level is elevated",
                    "action": "Consider reducing exposure to high-volatility positions"
                })
            
            return recommendations
            
        except Exception as e:
            return [{"type": "error", "message": str(e)}]
    
    def _calculate_diversification_score(self, allocation: Dict[str, float]) -> float:
        """Calculate portfolio diversification score (0-1)"""
        try:
            # Simple diversification score based on number of positions and concentration
            num_positions = len(allocation)
            if num_positions <= 1:
                return 0.0
            
            # Herfindahl-Hirschman Index for concentration
            hhi = sum(pct ** 2 for pct in allocation.values())
            # Normalize to 0-1 scale (1 = perfectly diversified, 0 = completely concentrated)
            diversification_score = 1 - hhi
            
            # Boost score for having more positions
            position_bonus = min(num_positions / 10, 0.3)  # Max 30% bonus for 10+ positions
            
            return min(diversification_score + position_bonus, 1.0)
            
        except Exception:
            return 0.5  # Default moderate diversification
    
    def _simulate_risk_adjusted_return(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Simulate risk-adjusted return metrics"""
        try:
            # Simulated metrics based on allocation characteristics
            total_allocation = sum(allocation.values())
            
            # Calculate weighted "risk" score
            weighted_risk = sum(allocation.get(symbol, 0) * 0.8 for symbol in allocation.keys())
            
            # Simulated returns based on diversification
            diversification_score = self._calculate_diversification_score(allocation)
            simulated_return = 0.08 + (diversification_score * 0.04)  # 8-12% range
            simulated_volatility = weighted_risk * 0.25  # Risk-adjusted volatility
            sharpe_ratio = (simulated_return - 0.02) / max(simulated_volatility, 0.01)  # Risk-free rate 2%
            
            return {
                "expected_return": round(simulated_return, 4),
                "expected_volatility": round(simulated_volatility, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
                "risk_adjusted_return": round(simulated_return / max(simulated_volatility, 0.01), 4)
            }
            
        except Exception:
            return {
                "expected_return": 0.10,
                "expected_volatility": 0.20,
                "sharpe_ratio": 0.40,
                "risk_adjusted_return": 0.50
            }
    
    def _estimate_monthly_cash_flow(self, symbols: List[str], allocation: Dict[str, float]) -> float:
        """Estimate monthly cash flow from portfolio"""
        try:
            # Simulated monthly cash flow estimate
            base_monthly_flow = len(symbols) * 50  # $50 per position base
            diversification_factor = min(len(symbols) / 5, 2.0)  # Up to 2x bonus for diversification
            concentration_penalty = max(allocation.values()) * 0.5  # Penalty for concentration
            
            estimated_flow = (base_monthly_flow * diversification_factor) - (concentration_penalty * 100)
            return max(estimated_flow, 0)  # Non-negative cash flow
            
        except Exception:
            return 500.0  # Default estimate
    
    def _analyze_concentration(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze concentration risk in portfolio"""
        try:
            sorted_allocations = sorted(allocation.values(), reverse=True)
            
            return {
                "max_allocation": max(allocation.values()),
                "top_3_concentration": sum(sorted_allocations[:3]),
                "concentration_ratio": max(allocation.values()) / sum(allocation.values()),
                "risk_level": "high" if max(allocation.values()) > 0.30 else "medium" if max(allocation.values()) > 0.20 else "low"
            }
            
        except Exception:
            return {"max_allocation": 0.25, "concentration_ratio": 0.25, "risk_level": "medium"}
    
    def _analyze_sector_balance(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze sector balance (simplified)"""
        # This would normally use real sector data
        # For now, return simulated analysis
        return {
            "sector_diversity": "moderate",
            "dominant_sectors": symbols[:3] if len(symbols) >= 3 else symbols,
            "recommended_additions": ["REITs", "Commodities", "International"],
            "balance_score": 0.7
        }
    
    def _generate_cash_flow_recommendations(self, allocation: Dict[str, float]) -> List[str]:
        """Generate cash flow optimization recommendations"""
        recommendations = []
        
        try:
            max_allocation = max(allocation.values())
            if max_allocation > 0.25:
                recommendations.append("Consider reducing position concentration to improve risk management")
            
            if len(allocation) < 5:
                recommendations.append("Portfolio could benefit from additional diversification")
            
            recommendations.append("Monitor quarterly rebalancing opportunities")
            recommendations.append("Consider tax-loss harvesting during market downturns")
            
            return recommendations
            
        except Exception:
            return ["Monitor portfolio performance regularly"]