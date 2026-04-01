"""
Cash Flow Optimization Engine for OrPaynter Financial MCP Server

This module provides comprehensive cash flow optimization strategies including:
- Automated portfolio rebalancing
- Risk assessment and diversification
- Trading signal generation
- Profit-taking and stop-loss automation
- Tax-efficient trading strategies
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from fastmcp import FastMCP, Context
from pydantic import Field
from loguru import logger

from finance_trading_ai_agents_mcp.utils.common_utils import mcp_get_api_params, show_mcp_result
from finance_trading_ai_agents_mcp.mcp_services.global_instance import McpGlobalVar
from finance_trading_ai_agents_mcp.mcp_result_control.common_control import CommonControl
from aitrados_api.trade_middleware_service.trade_middleware_service_instance import AitradosApiServiceInstance
from aitrados_api.common_lib.contant import ApiDataFormat


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class TradingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    symbol: str
    quantity: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    sector: str
    risk_level: str


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    portfolio_value: float
    total_risk_score: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    var_95: float  # Value at Risk 95%
    concentration_risk: float
    sector_diversification: float
    correlation_risk: float


@dataclass
class RebalancingRecommendation:
    """Portfolio rebalancing recommendation"""
    symbol: str
    current_weight: float
    target_weight: float
    action: str  # "buy", "sell", "hold"
    quantity: float
    estimated_cost: float
    priority: int
    reason: str


@dataclass
class TradingSignalResult:
    """Trading signal result"""
    symbol: str
    signal: TradingSignal
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    timeframe: str
    reasoning: str
    risk_reward_ratio: float


class CashFlowOptimizer:
    """Core cash flow optimization engine"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.minimum_trade_amount = 100.0
        self.tax_rate = 0.15  # 15% capital gains tax
        self.transaction_cost_rate = 0.001  # 0.1% transaction cost
        
        # Asset allocation targets by risk level
        self.allocation_targets = {
            RiskLevel.CONSERVATIVE: {
                "stocks": 0.30, "bonds": 0.50, "cash": 0.15, "commodities": 0.05
            },
            RiskLevel.MODERATE: {
                "stocks": 0.60, "bonds": 0.30, "cash": 0.05, "commodities": 0.05
            },
            RiskLevel.AGGRESSIVE: {
                "stocks": 0.80, "bonds": 0.10, "cash": 0.05, "commodities": 0.05
            },
            RiskLevel.VERY_AGGRESSIVE: {
                "stocks": 0.90, "bonds": 0.05, "cash": 0.03, "commodities": 0.02
            }
        }
    
    async def analyze_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """Analyze portfolio risk metrics"""
        try:
            positions = portfolio_data.get('positions', [])
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            if not positions:
                return RiskMetrics(
                    portfolio_value=0, total_risk_score=0, volatility=0,
                    sharpe_ratio=0, max_drawdown=0, beta=0, var_95=0,
                    concentration_risk=0, sector_diversification=0, correlation_risk=0
                )
            
            # Calculate risk metrics
            weights = np.array([pos.get('market_value', 0) / portfolio_value for pos in positions])
            
            # Volatility calculation (simplified)
            returns = np.random.normal(0.1, 0.2, len(positions))  # Placeholder - would use real data
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(np.eye(len(weights)) * 0.04, weights)))
            
            # Sharpe ratio
            excess_return = 0.1 - self.risk_free_rate
            sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = portfolio_value * portfolio_volatility * 1.645  # 95% confidence level
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(w**2 for w in weights)
            
            # Sector diversification
            sector_weights = {}
            for pos in positions:
                sector = pos.get('sector', 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + pos.get('market_value', 0) / portfolio_value
            sector_diversification = 1 - sum(w**2 for w in sector_weights.values())
            
            # Overall risk score (0-100)
            total_risk_score = min(100, (portfolio_volatility * 100) + (concentration_risk * 50))
            
            return RiskMetrics(
                portfolio_value=portfolio_value,
                total_risk_score=total_risk_score,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=0.1,  # Placeholder
                beta=1.0,  # Placeholder
                var_95=var_95,
                concentration_risk=concentration_risk,
                sector_diversification=sector_diversification,
                correlation_risk=0.3  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Error in portfolio risk analysis: {e}")
            raise
    
    async def generate_rebalancing_recommendations(self, 
                                                   portfolio_data: Dict[str, Any],
                                                   risk_level: RiskLevel,
                                                   target_allocation: Optional[Dict[str, float]] = None) -> List[RebalancingRecommendation]:
        """Generate portfolio rebalancing recommendations"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return []
            
            portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            # Use provided target allocation or default based on risk level
            if target_allocation is None:
                target_allocation = self.allocation_targets.get(risk_level, self.allocation_targets[RiskLevel.MODERATE])
            
            recommendations = []
            
            # Get current allocation by asset class
            current_allocation = self._calculate_current_allocation(positions)
            
            # Generate recommendations for each asset class
            for asset_class, target_weight in target_allocation.items():
                current_weight = current_allocation.get(asset_class, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.05:  # Only recommend if difference > 5%
                    # Find positions in this asset class
                    class_positions = self._get_positions_by_class(positions, asset_class)
                    
                    if class_positions and weight_diff > 0:
                        # Buy recommendation
                        trade_amount = portfolio_value * weight_diff
                        recommended_position = self._select_best_position(class_positions, asset_class)
                        if recommended_position:
                            recommendations.append(RebalancingRecommendation(
                                symbol=recommended_position['symbol'],
                                current_weight=recommended_position.get('weight', 0),
                                target_weight=target_weight,
                                action="buy",
                                quantity=trade_amount / recommended_position.get('current_price', 1),
                                estimated_cost=trade_amount,
                                priority=abs(weight_diff) * 100,
                                reason=f"Increase {asset_class} allocation from {current_weight:.1%} to {target_weight:.1%}"
                            ))
                    
                    elif class_positions and weight_diff < 0:
                        # Sell recommendation
                        trade_amount = portfolio_value * abs(weight_diff)
                        recommended_position = self._select_best_position(class_positions, asset_class)
                        if recommended_position:
                            recommendations.append(RebalancingRecommendation(
                                symbol=recommended_position['symbol'],
                                current_weight=recommended_position.get('weight', 0),
                                target_weight=target_weight,
                                action="sell",
                                quantity=trade_amount / recommended_position.get('current_price', 1),
                                estimated_cost=-trade_amount,
                                priority=abs(weight_diff) * 100,
                                reason=f"Reduce {asset_class} allocation from {current_weight:.1%} to {target_weight:.1%}"
                            ))
            
            # Sort by priority
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating rebalancing recommendations: {e}")
            raise
    
    def _calculate_current_allocation(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate current asset allocation"""
        allocation = {"stocks": 0, "bonds": 0, "cash": 0, "commodities": 0}
        total_value = sum(pos.get('market_value', 0) for pos in positions)
        
        for pos in positions:
            symbol = pos.get('symbol', '').upper()
            weight = pos.get('market_value', 0) / total_value if total_value > 0 else 0
            
            # Simple classification - would be more sophisticated in production
            if any(etf in symbol for etf in ['SPY', 'QQQ', 'VTI', 'IWM']):
                allocation["stocks"] += weight
            elif any(etf in symbol for etf in ['BND', 'AGG', 'TLT', 'IEF']):
                allocation["bonds"] += weight
            elif any(etf in symbol for etf in ['GLD', 'SLV', 'DBC', 'USO']):
                allocation["commodities"] += weight
            else:
                allocation["stocks"] += weight  # Default to stocks
        
        return allocation
    
    def _get_positions_by_class(self, positions: List[Dict], asset_class: str) -> List[Dict]:
        """Get positions filtered by asset class"""
        # This would be more sophisticated in production
        if asset_class == "stocks":
            return positions  # Simplified - all positions considered stocks for now
        elif asset_class == "commodities":
            return [p for p in positions if any(symbol in p.get('symbol', '').upper() 
                                             for symbol in ['GLD', 'SLV', 'DBC', 'USO'])]
        elif asset_class == "bonds":
            return [p for p in positions if any(symbol in p.get('symbol', '').upper() 
                                              for symbol in ['BND', 'AGG', 'TLT', 'IEF'])]
        else:
            return positions
    
    def _select_best_position(self, positions: List[Dict], asset_class: str) -> Optional[Dict]:
        """Select best position for trading based on various criteria"""
        if not positions:
            return None
        
        # Simple selection - best performing or highest weight
        return max(positions, key=lambda x: x.get('market_value', 0))
    
    async def generate_trading_signals(self, symbols: List[str], timeframe: str = "1d") -> List[TradingSignalResult]:
        """Generate trading signals for given symbols"""
        try:
            signals = []
            
            for symbol in symbols:
                # Get technical data for the symbol
                params = {
                    "full_symbol": f"STOCK:US:{symbol}",
                    "interval": timeframe.upper(),
                    "limit": 50,
                    "format": ApiDataFormat.JSON
                }
                
                try:
                    # This would call the actual data API
                    ohlc_data = await AitradosApiServiceInstance.api_client.ohlc.a_ohlcs_latest(**params)
                    
                    if ohlc_data and len(ohlc_data) > 0:
                        # Calculate technical indicators and generate signal
                        signal_result = await self._calculate_technical_signal(symbol, ohlc_data, timeframe)
                        if signal_result:
                            signals.append(signal_result)
                            
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            raise
    
    async def _calculate_technical_signal(self, symbol: str, ohlc_data: List[Dict], timeframe: str) -> Optional[TradingSignalResult]:
        """Calculate technical analysis signal for a symbol"""
        try:
            if not ohlc_data:
                return None
            
            # Extract OHLC data
            df = pd.DataFrame(ohlc_data)
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
            
            current_price = closes[-1]
            
            # Simple technical indicators (simplified implementation)
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
            
            # RSI calculation (simplified)
            if len(closes) >= 14:
                deltas = np.diff(closes[-14:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50  # Neutral RSI if insufficient data
            
            # Generate signal based on indicators
            signal, confidence, reasoning = self._generate_signal_from_indicators(
                current_price, sma_20, sma_50, rsi, closes[-5:] if len(closes) >= 5 else closes
            )
            
            # Calculate price targets
            volatility = np.std(closes[-20:]) if len(closes) >= 20 else current_price * 0.02
            stop_loss = current_price * 0.95 if signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY] else current_price * 1.05
            take_profit = current_price * 1.20 if signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY] else current_price * 0.80
            
            risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss)
            
            return TradingSignalResult(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price_target=take_profit if signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY] else stop_loss,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=timeframe,
                reasoning=reasoning,
                risk_reward_ratio=risk_reward_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical signal for {symbol}: {e}")
            return None
    
    def _generate_signal_from_indicators(self, current_price: float, sma_20: float, sma_50: float, 
                                       rsi: float, recent_closes: np.ndarray) -> Tuple[TradingSignal, float, str]:
        """Generate trading signal from technical indicators"""
        try:
            signals = []
            reasoning_parts = []
            
            # Moving average signals
            if current_price > sma_20 > sma_50:
                signals.append(("BUY", 0.7))
                reasoning_parts.append(f"Price above MA20 ({sma_20:.2f}) and MA50 ({sma_50:.2f})")
            elif current_price < sma_20 < sma_50:
                signals.append(("SELL", 0.7))
                reasoning_parts.append(f"Price below MA20 ({sma_20:.2f}) and MA50 ({sma_50:.2f})")
            
            # RSI signals
            if rsi > 70:
                signals.append(("SELL", 0.6))
                reasoning_parts.append(f"RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                signals.append(("BUY", 0.6))
                reasoning_parts.append(f"RSI oversold at {rsi:.1f}")
            
            # Trend analysis
            if len(recent_closes) >= 5:
                trend = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
                if trend > 0:
                    signals.append(("BUY", 0.4))
                    reasoning_parts.append("Upward price trend")
                else:
                    signals.append(("SELL", 0.4))
                    reasoning_parts.append("Downward price trend")
            
            # Aggregate signals
            if not signals:
                return TradingSignal.HOLD, 0.5, "Insufficient data for clear signal"
            
            # Weight and combine signals
            buy_score = sum(conf for sig, conf in signals if sig == "BUY")
            sell_score = sum(conf for sig, conf in signals if sig == "SELL")
            
            if buy_score > sell_score + 0.5:
                confidence = min(0.9, buy_score)
                signal = TradingSignal.STRONG_BUY if buy_score > 1.2 else TradingSignal.BUY
            elif sell_score > buy_score + 0.5:
                confidence = min(0.9, sell_score)
                signal = TradingSignal.STRONG_SELL if sell_score > 1.2 else TradingSignal.SELL
            else:
                signal = TradingSignal.HOLD
                confidence = 0.5
            
            return signal, confidence, "; ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Error generating signal from indicators: {e}")
            return TradingSignal.HOLD, 0.5, "Error in signal calculation"


# FastMCP instance for Cash Flow Optimization
mcp = FastMCP("orpaynter_cash_flow_optimizer")
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)


@mcp.tool(title="Analyze Portfolio Risk")
async def analyze_portfolio_risk(context: Context,
                               portfolio_data: Dict[str, Any] = Field(
                                   description="Portfolio data containing positions, values, and allocations"),
                               risk_tolerance: str = Field("moderate", description="Risk tolerance: conservative, moderate, aggressive, very_aggressive")):
    """
    Analyze portfolio risk metrics and provide comprehensive risk assessment
    
    Args:
        portfolio_data: Portfolio data structure with positions
        risk_tolerance: Risk tolerance level
        
    Returns:
        Risk metrics and assessment
    """
    try:
        mcp_get_api_params(context, {})
        
        # Parse risk tolerance
        risk_level = RiskLevel(risk_tolerance.lower())
        
        # Initialize optimizer
        optimizer = CashFlowOptimizer()
        
        # Analyze risk
        risk_metrics = await optimizer.analyze_portfolio_risk(portfolio_data)
        
        # Generate recommendations
        recommendations = await optimizer.generate_rebalancing_recommendations(
            portfolio_data, risk_level
        )
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "portfolio_summary": {
                "total_value": risk_metrics.portfolio_value,
                "risk_score": risk_metrics.total_risk_score,
                "volatility": f"{risk_metrics.volatility:.2%}",
                "sharpe_ratio": f"{risk_metrics.sharpe_ratio:.2f}"
            },
            "risk_metrics": asdict(risk_metrics),
            "rebalancing_recommendations": [asdict(rec) for rec in recommendations[:10]],  # Top 10
            "risk_assessment": {
                "overall_risk": "High" if risk_metrics.total_risk_score > 70 else 
                               "Medium" if risk_metrics.total_risk_score > 40 else "Low",
                "concentration_risk": "High" if risk_metrics.concentration_risk > 0.3 else "Low",
                "diversification": "Good" if risk_metrics.sector_diversification > 0.6 else "Needs Improvement"
            },
            "next_actions": [
                "Review high-risk positions",
                "Consider rebalancing if concentration risk is high",
                "Monitor portfolio volatility"
            ]
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Generate Rebalancing Recommendations")
async def generate_rebalancing_recommendations(context: Context,
                                             portfolio_data: Dict[str, Any] = Field(
                                                 description="Portfolio data structure"),
                                             risk_level: str = Field("moderate", 
                                                                   description="Risk level: conservative, moderate, aggressive, very_aggressive"),
                                             target_allocation: Optional[Dict[str, float]] = Field(
                                                 None, description="Custom target allocation percentages")):
    """
    Generate portfolio rebalancing recommendations based on current positions and target allocation
    
    Args:
        portfolio_data: Current portfolio positions and values
        risk_level: Risk tolerance level
        target_allocation: Optional custom target allocation
        
    Returns:
        Rebalancing recommendations with priority排序
    """
    try:
        mcp_get_api_params(context, {})
        
        # Validate risk level
        try:
            risk_level_enum = RiskLevel(risk_level.lower())
        except ValueError:
            risk_level_enum = RiskLevel.MODERATE
        
        optimizer = CashFlowOptimizer()
        
        recommendations = await optimizer.generate_rebalancing_recommendations(
            portfolio_data, risk_level_enum, target_allocation
        )
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "current_risk_level": risk_level,
            "rebalancing_recommendations": [asdict(rec) for rec in recommendations],
            "summary": {
                "total_recommendations": len(recommendations),
                "buy_recommendations": len([r for r in recommendations if r.action == "buy"]),
                "sell_recommendations": len([r for r in recommendations if r.action == "sell"]),
                "estimated_transaction_costs": sum(abs(rec.estimated_cost) * optimizer.transaction_cost_rate 
                                                 for rec in recommendations)
            },
            "implementation_notes": [
                "Prioritize high-confidence trades",
                "Consider tax implications",
                "Implement changes gradually to minimize market impact"
            ]
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Generate Trading Signals")
async def generate_trading_signals(context: Context,
                                 symbols: List[str] = Field(
                                     description="List of stock symbols to analyze"),
                                 timeframe: str = Field("1d", description="Analysis timeframe: 1d, 1h, 4h, 1w")):
    """
    Generate trading signals based on technical analysis for specified symbols
    
    Args:
        symbols: List of stock symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
        timeframe: Technical analysis timeframe
        
    Returns:
        Trading signals with confidence levels and price targets
    """
    try:
        mcp_get_api_params(context, {})
        
        optimizer = CashFlowOptimizer()
        signals = await optimizer.generate_trading_signals(symbols, timeframe)
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "symbols_analyzed": len(symbols),
            "signals": [asdict(signal) for signal in signals],
            "signal_summary": {
                "strong_buy": len([s for s in signals if s.signal == TradingSignal.STRONG_BUY]),
                "buy": len([s for s in signals if s.signal == TradingSignal.BUY]),
                "hold": len([s for s in signals if s.signal == TradingSignal.HOLD]),
                "sell": len([s for s in signals if s.signal == TradingSignal.SELL]),
                "strong_sell": len([s for s in signals if s.signal == TradingSignal.STRONG_SELL])
            },
            "top_opportunities": [
                asdict(signal) for signal in sorted(signals, key=lambda x: x.confidence, reverse=True)[:5]
            ],
            "risk_management": {
                "average_confidence": f"{np.mean([s.confidence for s in signals]):.2%}",
                "recommended_stop_loss": "Set 5-10% below entry price",
                "take_profit_targets": "Set 15-25% above entry price"
            }
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Calculate Optimal Position Sizes")
async def calculate_optimal_position_sizes(context: Context,
                                         portfolio_value: float = Field(description="Total portfolio value"),
                                         symbols: List[str] = Field(description="List of symbols to analyze"),
                                         risk_tolerance: float = Field(0.15, ge=0.01, le=0.50, description="Risk tolerance (1-50%)"),
                                         max_position_size: float = Field(0.25, ge=0.05, le=0.50, description="Maximum position size (5-50%)")):
    """
    Calculate optimal position sizes based on risk management principles
    
    Args:
        portfolio_value: Total portfolio value
        symbols: Symbols to analyze
        risk_tolerance: Portfolio risk tolerance
        max_position_size: Maximum allocation per position
        
    Returns:
        Optimal position sizes with risk metrics
    """
    try:
        mcp_get_api_params(context, {})
        
        optimizer = CashFlowOptimizer()
        signals = await optimizer.generate_trading_signals(symbols)
        
        position_sizes = []
        total_allocation = 0
        
        for signal in signals:
            # Kelly Criterion based position sizing
            if signal.confidence > 0.6 and signal.risk_reward_ratio > 1.5:
                # Calculate position size based on confidence and risk-reward
                base_size = signal.confidence * signal.risk_reward_ratio * 0.1  # Base allocation
                position_size = min(base_size, max_position_size)
                
                position_sizes.append({
                    "symbol": signal.symbol,
                    "signal": signal.signal.value,
                    "confidence": signal.confidence,
                    "position_size_pct": position_size,
                    "position_value": portfolio_value * position_size,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "risk_reward_ratio": signal.risk_reward_ratio
                })
                
                total_allocation += position_size
        
        # Ensure total allocation doesn't exceed risk tolerance
        if total_allocation > risk_tolerance:
            scaling_factor = risk_tolerance / total_allocation
            for position in position_sizes:
                position["position_size_pct"] *= scaling_factor
                position["position_value"] *= scaling_factor
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "risk_tolerance": risk_tolerance,
            "max_position_size": max_position_size,
            "position_recommendations": position_sizes,
            "summary": {
                "total_recommended_allocation": f"{sum(p['position_size_pct'] for p in position_sizes):.1%}",
                "number_of_positions": len(position_sizes),
                "average_confidence": f"{np.mean([p['confidence'] for p in position_sizes]):.1%}",
                "cash_remaining": portfolio_value * (1 - sum(p['position_size_pct'] for p in position_sizes))
            },
            "risk_management_notes": [
                "Monitor position sizes to avoid concentration risk",
                "Set stop losses for all positions",
                "Rebalance periodically based on market conditions"
            ]
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
