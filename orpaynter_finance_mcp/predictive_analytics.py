"""
Predictive Analytics Service for OrPaynter Financial MCP Server

Provides advanced predictive analytics capabilities including:
- Market trend prediction using AI models
- Volatility forecasting
- Correlation analysis across asset classes
- Economic indicator integration
- Pattern recognition and anomaly detection
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
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from fastmcp import FastMCP, Context
from pydantic import Field
from loguru import logger

from finance_trading_ai_agents_mcp.utils.common_utils import mcp_get_api_params, show_mcp_result
from finance_trading_ai_agents_mcp.mcp_services.global_instance import McpGlobalVar
from finance_trading_ai_agents_mcp.mcp_result_control.common_control import CommonControl
from aitrados_api.trade_middleware_service.trade_middleware_service_instance import AitradosApiServiceInstance
from aitrados_api.common_lib.contant import ApiDataFormat


class PredictionHorizon(Enum):
    SHORT_TERM = "1d"      # 1 day
    MEDIUM_TERM = "1w"     # 1 week
    LONG_TERM = "1m"       # 1 month
    EXTENDED_TERM = "3m"   # 3 months


class TrendDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class AnomalyType(Enum):
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    UNUSUAL_VOLATILITY = "unusual_volatility"
    CORRELATION_BREAK = "correlation_break"
    PATTERN_BREAK = "pattern_break"


@dataclass
class PricePrediction:
    """Price prediction result"""
    symbol: str
    current_price: float
    predicted_price: float
    confidence: float
    prediction_horizon: str
    trend_direction: TrendDirection
    potential_return: float
    volatility_forecast: float
    model_accuracy: float
    key_factors: List[str]
    timestamp: datetime


@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    symbol: str
    current_volatility: float
    predicted_volatility: float
    forecast_horizon: str
    confidence: float
    volatility_regime: str  # low, normal, high, extreme
    factors_affecting_volatility: List[str]
    trading_implications: List[str]


@dataclass
class CorrelationAnalysis:
    """Correlation analysis result"""
    asset_pairs: List[str]
    current_correlation: float
    predicted_correlation: float
    correlation_trend: str  # increasing, decreasing, stable
    diversification_benefit: float
    risk_implications: List[str]
    rebalancing_suggestions: List[str]


@dataclass
class MarketAnomaly:
    """Market anomaly detection result"""
    anomaly_id: str
    symbol: str
    anomaly_type: AnomalyType
    severity: str  # low, medium, high, critical
    description: str
    detected_value: float
    expected_range: Tuple[float, float]
    confidence: float
    recommended_actions: List[str]
    timestamp: datetime


@dataclass
class EconomicIndicator:
    """Economic indicator data"""
    indicator_name: str
    current_value: float
    previous_value: float
    change_percentage: float
    forecast: float
    impact_on_market: str
    relevance_score: float
    last_updated: datetime


class PredictiveAnalytics:
    """Core predictive analytics engine"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Economic indicators to monitor
        self.economic_indicators = {
            "federal_funds_rate": {"weight": 0.3, "category": "monetary_policy"},
            "inflation_rate": {"weight": 0.25, "category": "economic_growth"},
            "unemployment_rate": {"weight": 0.2, "category": "labor_market"},
            "gdp_growth": {"weight": 0.15, "category": "economic_growth"},
            "vix": {"weight": 0.1, "category": "market_sentiment"}
        }
        
        # Prediction models configuration
        self.model_config = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "feature_window": 30,  # Days of historical data for features
            "prediction_horizons": [1, 5, 10, 20]  # Days ahead
        }
    
    async def predict_price_trend(self, symbol: str, 
                                horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM) -> Optional[PricePrediction]:
        """Predict price trend for a symbol"""
        try:
            # Get historical data
            historical_data = await self._get_historical_data(symbol, days=90)
            
            if not historical_data or len(historical_data) < 30:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Prepare features
            features = self._prepare_features(historical_data)
            target = self._prepare_target(historical_data, horizon)
            
            if features is None or target is None:
                return None
            
            # Train model
            model = RandomForestRegressor(**self.model_config["random_forest"])
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = r2_score(y_test, y_pred)
            
            # Make prediction
            latest_features = features[-1:].reshape(1, -1)
            latest_features_scaled = scaler.transform(latest_features)
            predicted_price = model.predict(latest_features_scaled)[0]
            
            current_price = historical_data[-1]['close']
            
            # Calculate confidence and trend
            confidence = min(0.95, accuracy)
            trend_direction = self._determine_trend_direction(current_price, predicted_price)
            potential_return = (predicted_price - current_price) / current_price
            
            # Volatility forecast
            volatility_forecast = self._forecast_volatility(historical_data, horizon)
            
            # Key factors analysis
            key_factors = self._identify_key_factors(model, historical_data)
            
            return PricePrediction(
                symbol=symbol,
                current_price=current_price,
                predicted_price=predicted_price,
                confidence=confidence,
                prediction_horizon=horizon.value,
                trend_direction=trend_direction,
                potential_return=potential_return,
                volatility_forecast=volatility_forecast,
                model_accuracy=accuracy,
                key_factors=key_factors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting price trend for {symbol}: {e}")
            return None
    
    async def forecast_volatility(self, symbol: str, 
                                horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM) -> Optional[VolatilityForecast]:
        """Forecast volatility for a symbol"""
        try:
            # Get historical data
            historical_data = await self._get_historical_data(symbol, days=60)
            
            if not historical_data or len(historical_data) < 20:
                return None
            
            # Calculate current volatility
            returns = [data['return'] for data in historical_data if 'return' in data]
            current_volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Volatility regime classification
            volatility_regime = self._classify_volatility_regime(current_volatility)
            
            # Forecast future volatility (simplified)
            predicted_volatility = self._forecast_volatility(historical_data, horizon)
            
            # Factors affecting volatility
            factors = self._identify_volatility_factors(historical_data)
            
            # Trading implications
            implications = self._get_volatility_implications(volatility_regime, predicted_volatility)
            
            return VolatilityForecast(
                symbol=symbol,
                current_volatility=current_volatility,
                predicted_volatility=predicted_volatility,
                forecast_horizon=horizon.value,
                confidence=0.75,  # Volatility forecasting confidence
                volatility_regime=volatility_regime,
                factors_affecting_volatility=factors,
                trading_implications=implications
            )
            
        except Exception as e:
            logger.error(f"Error forecasting volatility for {symbol}: {e}")
            return None
    
    async def analyze_correlations(self, symbols: List[str]) -> List[CorrelationAnalysis]:
        """Analyze correlations between asset pairs"""
        try:
            correlations = []
            
            # Get historical data for all symbols
            historical_data = {}
            for symbol in symbols:
                data = await self._get_historical_data(symbol, days=90)
                if data and len(data) > 20:
                    historical_data[symbol] = data
            
            if len(historical_data) < 2:
                return []
            
            # Calculate correlations between all pairs
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in historical_data and symbol2 in historical_data:
                        correlation_result = await self._calculate_correlation(
                            symbol1, symbol2, historical_data[symbol1], historical_data[symbol2]
                        )
                        if correlation_result:
                            correlations.append(correlation_result)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return []
    
    async def detect_anomalies(self, symbols: List[str]) -> List[MarketAnomaly]:
        """Detect market anomalies"""
        try:
            anomalies = []
            
            for symbol in symbols:
                # Get recent data
                recent_data = await self._get_historical_data(symbol, days=30)
                
                if recent_data and len(recent_data) > 10:
                    symbol_anomalies = await self._detect_symbol_anomalies(symbol, recent_data)
                    anomalies.extend(symbol_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def get_economic_indicators(self) -> List[EconomicIndicator]:
        """Get and forecast economic indicators"""
        try:
            indicators = []
            
            # Simulate economic indicators (in production, would fetch from APIs)
            indicator_data = {
                "federal_funds_rate": {"current": 5.25, "previous": 5.00, "forecast": 5.25},
                "inflation_rate": {"current": 3.2, "previous": 3.7, "forecast": 2.8},
                "unemployment_rate": {"current": 3.7, "previous": 3.8, "forecast": 3.6},
                "gdp_growth": {"current": 2.1, "previous": 2.9, "forecast": 1.8},
                "vix": {"current": 18.5, "previous": 22.1, "forecast": 16.8}
            }
            
            for name, data in indicator_data.items():
                change_pct = ((data["current"] - data["previous"]) / data["previous"]) * 100
                
                indicator = EconomicIndicator(
                    indicator_name=name.replace("_", " ").title(),
                    current_value=data["current"],
                    previous_value=data["previous"],
                    change_percentage=change_pct,
                    forecast=data["forecast"],
                    impact_on_market=self._assess_indicator_impact(name, data),
                    relevance_score=self.economic_indicators.get(name, {}).get("weight", 0.1),
                    last_updated=datetime.now()
                )
                indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return []
    
    async def _get_historical_data(self, symbol: str, days: int) -> List[Dict]:
        """Get historical price data for a symbol"""
        try:
            params = {
                "full_symbol": f"STOCK:US:{symbol}",
                "interval": "DAY",
                "limit": days,
                "format": ApiDataFormat.JSON
            }
            
            ohlc_data = await AitradosApiServiceInstance.api_client.ohlc.a_ohlcs_latest(**params)
            
            if not ohlc_data:
                return []
            
            # Process data and add technical indicators
            processed_data = []
            for i, data in enumerate(ohlc_data):
                processed_point = {
                    "date": data.get("close_datetime", ""),
                    "open": float(data.get("open", 0)),
                    "high": float(data.get("high", 0)),
                    "low": float(data.get("low", 0)),
                    "close": float(data.get("close", 0)),
                    "volume": float(data.get("volume", 0))
                }
                
                # Add return calculation
                if i > 0:
                    prev_close = ohlc_data[i-1].get("close", 0)
                    if prev_close > 0:
                        processed_point["return"] = (processed_point["close"] - prev_close) / prev_close
                    else:
                        processed_point["return"] = 0
                else:
                    processed_point["return"] = 0
                
                processed_data.append(processed_point)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def _prepare_features(self, historical_data: List[Dict]) -> Optional[np.ndarray]:
        """Prepare features for machine learning model"""
        try:
            if len(historical_data) < self.model_config["feature_window"]:
                return None
            
            features = []
            prices = [data["close"] for data in historical_data]
            volumes = [data["volume"] for data in historical_data]
            returns = [data.get("return", 0) for data in historical_data]
            
            # Create feature matrix
            for i in range(self.model_config["feature_window"], len(historical_data)):
                feature_vector = []
                
                # Price-based features
                current_price = prices[i]
                
                # Moving averages
                sma_5 = np.mean(prices[i-5:i]) if i >= 5 else current_price
                sma_10 = np.mean(prices[i-10:i]) if i >= 10 else current_price
                sma_20 = np.mean(prices[i-20:i]) if i >= 20 else current_price
                
                # Price ratios
                feature_vector.extend([
                    current_price / sma_5 - 1,  # Price vs SMA5
                    current_price / sma_10 - 1,  # Price vs SMA10
                    current_price / sma_20 - 1,  # Price vs SMA20
                    sma_5 / sma_10 - 1,  # SMA5 vs SMA10
                    sma_10 / sma_20 - 1,  # SMA10 vs SMA20
                ])
                
                # Volatility features
                recent_returns = returns[i-10:i] if i >= 10 else returns[:i]
                volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
                feature_vector.append(volatility)
                
                # Volume features
                avg_volume = np.mean(volumes[i-5:i]) if i >= 5 else volumes[i] if volumes else 1
                volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1
                feature_vector.append(volume_ratio)
                
                # Momentum features
                momentum_5 = (prices[i] / prices[i-5] - 1) if i >= 5 else 0
                momentum_10 = (prices[i] / prices[i-10] - 1) if i >= 10 else 0
                feature_vector.extend([momentum_5, momentum_10])
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _prepare_target(self, historical_data: List[Dict], horizon: PredictionHorizon) -> Optional[np.ndarray]:
        """Prepare target variable for prediction"""
        try:
            prices = [data["close"] for data in historical_data]
            horizon_days = {"1d": 1, "1w": 5, "1m": 20, "3m": 60}[horizon.value]
            
            targets = []
            for i in range(len(prices) - horizon_days):
                current_price = prices[i]
                future_price = prices[i + horizon_days]
                target = (future_price - current_price) / current_price
                targets.append(target)
            
            return np.array(targets)
            
        except Exception as e:
            logger.error(f"Error preparing target: {e}")
            return None
    
    def _determine_trend_direction(self, current_price: float, predicted_price: float) -> TrendDirection:
        """Determine trend direction based on price change"""
        change_pct = (predicted_price - current_price) / current_price
        
        if change_pct > 0.05:
            return TrendDirection.STRONG_BULLISH
        elif change_pct > 0.02:
            return TrendDirection.BULLISH
        elif change_pct < -0.05:
            return TrendDirection.STRONG_BEARISH
        elif change_pct < -0.02:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL
    
    def _forecast_volatility(self, historical_data: List[Dict], horizon: PredictionHorizon) -> float:
        """Forecast volatility for the given horizon"""
        returns = [data.get("return", 0) for data in historical_data if "return" in data]
        
        if len(returns) < 10:
            return 0.20  # Default volatility
        
        # Calculate historical volatility
        hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Simple volatility forecast (could be enhanced with GARCH models)
        volatility_factor = {"1d": 1.0, "1w": 1.1, "1m": 1.2, "3m": 1.3}[horizon.value]
        forecast_vol = hist_vol * volatility_factor
        
        return min(forecast_vol, 1.0)  # Cap at 100%
    
    def _identify_key_factors(self, model, historical_data: List[Dict]) -> List[str]:
        """Identify key factors affecting price prediction"""
        factors = []
        
        # Technical factors
        if len(historical_data) >= 20:
            prices = [data["close"] for data in historical_data]
            recent_trend = prices[-1] / prices[-20] - 1
            if abs(recent_trend) > 0.05:
                factors.append(f"Strong recent trend ({recent_trend:.1%})")
        
        # Volume factors
        volumes = [data.get("volume", 0) for data in historical_data[-10:]]
        if volumes:
            avg_volume = np.mean(volumes[:-1])
            recent_volume = volumes[-1]
            if recent_volume > avg_volume * 1.5:
                factors.append("Unusual volume activity")
        
        # Market factors
        factors.extend([
            "Technical indicators momentum",
            "Market sentiment",
            "Economic conditions"
        ])
        
        return factors[:5]  # Return top 5 factors
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.15:
            return "Low"
        elif volatility < 0.25:
            return "Normal"
        elif volatility < 0.40:
            return "High"
        else:
            return "Extreme"
    
    def _identify_volatility_factors(self, historical_data: List[Dict]) -> List[str]:
        """Identify factors affecting volatility"""
        factors = []
        
        # Check for recent high volatility periods
        returns = [data.get("return", 0) for data in historical_data[-10:] if "return" in data]
        if returns:
            vol_10d = np.std(returns)
            vol_30d = np.std(returns[-30:]) if len(returns) >= 30 else vol_10d
            
            if vol_10d > vol_30d * 1.5:
                factors.append("Recent volatility spike")
        
        # Volume analysis
        volumes = [data.get("volume", 0) for data in historical_data[-5:]]
        if volumes:
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            if volume_trend > 0:
                factors.append("Increasing volume trend")
        
        factors.extend([
            "Market uncertainty",
            "Economic news flow",
            "Options expiration cycles"
        ])
        
        return factors
    
    def _get_volatility_implications(self, regime: str, forecast_vol: float) -> List[str]:
        """Get trading implications for volatility forecast"""
        implications = []
        
        if regime == "Low":
            implications.extend([
                "Consider volatility selling strategies",
                "Lower expected option premiums",
                "Reduced hedging costs"
            ])
        elif regime == "High":
            implications.extend([
                "Increased hedging costs",
                "Higher option premiums",
                "Consider volatility buying strategies"
            ])
        elif regime == "Extreme":
            implications.extend([
                "Market stress conditions",
                "Extreme caution required",
                "Consider defensive positioning"
            ])
        
        return implications
    
    async def _calculate_correlation(self, symbol1: str, symbol2: str, 
                                   data1: List[Dict], data2: List[Dict]) -> Optional[CorrelationAnalysis]:
        """Calculate correlation between two symbols"""
        try:
            # Align data by date
            returns1 = [data.get("return", 0) for data in data1 if "return" in data]
            returns2 = [data.get("return", 0) for data in data2 if "return" in data]
            
            min_length = min(len(returns1), len(returns2))
            if min_length < 20:
                return None
            
            returns1_aligned = returns1[-min_length:]
            returns2_aligned = returns2[-min_length:]
            
            # Calculate correlation
            correlation = np.corrcoef(returns1_aligned, returns2_aligned)[0, 1]
            
            # Predict future correlation (simplified)
            predicted_correlation = correlation * 0.95  # Slight mean reversion
            
            # Determine correlation trend
            correlation_trend = "stable"
            
            # Calculate diversification benefit
            diversification_benefit = 1 - correlation ** 2
            
            # Generate suggestions
            rebalancing_suggestions = []
            if correlation > 0.8:
                rebalancing_suggestions.append("High correlation - consider alternative assets")
            elif correlation < 0.3:
                rebalancing_suggestions.append("Low correlation - good diversification")
            
            return CorrelationAnalysis(
                asset_pairs=[symbol1, symbol2],
                current_correlation=correlation,
                predicted_correlation=predicted_correlation,
                correlation_trend=correlation_trend,
                diversification_benefit=diversification_benefit,
                risk_implications=[f"Correlation level: {correlation:.2f}"],
                rebalancing_suggestions=rebalancing_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None
    
    async def _detect_symbol_anomalies(self, symbol: str, recent_data: List[Dict]) -> List[MarketAnomaly]:
        """Detect anomalies for a specific symbol"""
        try:
            anomalies = []
            
            if len(recent_data) < 10:
                return anomalies
            
            # Volume spike detection
            volumes = [data.get("volume", 0) for data in recent_data]
            if volumes:
                avg_volume = np.mean(volumes[:-1])
                recent_volume = volumes[-1]
                
                if recent_volume > avg_volume * 3:  # 3x average volume
                    anomalies.append(MarketAnomaly(
                        anomaly_id=f"{symbol}_volume_spike_{datetime.now().timestamp()}",
                        symbol=symbol,
                        anomaly_type=AnomalyType.VOLUME_SPIKE,
                        severity="High" if recent_volume > avg_volume * 5 else "Medium",
                        description=f"Volume spike detected: {recent_volume/avg_volume:.1f}x average",
                        detected_value=recent_volume,
                        expected_range=(avg_volume * 0.5, avg_volume * 2),
                        confidence=0.8,
                        recommended_actions=["Monitor price action", "Check for news events"],
                        timestamp=datetime.now()
                    ))
            
            # Price gap detection
            closes = [data["close"] for data in recent_data]
            if len(closes) >= 2:
                prev_close = closes[-2]
                current_close = closes[-1]
                gap_size = abs(current_close - prev_close) / prev_close
                
                if gap_size > 0.05:  # 5% gap
                    anomalies.append(MarketAnomaly(
                        anomaly_id=f"{symbol}_price_gap_{datetime.now().timestamp()}",
                        symbol=symbol,
                        anomaly_type=AnomalyType.PRICE_GAP,
                        severity="High" if gap_size > 0.10 else "Medium",
                        description=f"Price gap detected: {gap_size:.1%}",
                        detected_value=gap_size,
                        expected_range=(0, 0.02),
                        confidence=0.9,
                        recommended_actions=["Investigate cause", "Monitor for reversal"],
                        timestamp=datetime.now()
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {symbol}: {e}")
            return []
    
    def _assess_indicator_impact(self, indicator_name: str, data: Dict) -> str:
        """Assess impact of economic indicator on markets"""
        current = data["current"]
        previous = data["previous"]
        change = (current - previous) / previous if previous != 0 else 0
        
        if abs(change) > 0.05:  # 5% change
            return "High Impact"
        elif abs(change) > 0.02:  # 2% change
            return "Medium Impact"
        else:
            return "Low Impact"


# FastMCP instance for Predictive Analytics
mcp = FastMCP("orpaynter_predictive_analytics")
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)


@mcp.tool(title="Predict Price Trends")
async def predict_price_trends(context: Context,
                             symbols: List[str] = Field(description="List of symbols to predict"),
                             prediction_horizon: str = Field("1w", description="Prediction horizon: 1d, 1w, 1m, 3m"),
                             confidence_threshold: float = Field(0.6, ge=0.5, le=0.95, description="Minimum confidence threshold")):
    """
    Predict price trends for specified symbols using machine learning models
    
    Args:
        symbols: List of stock symbols to analyze
        prediction_horizon: Time horizon for prediction
        confidence_threshold: Minimum confidence level for predictions
        
    Returns:
        Price predictions with confidence levels and trend analysis
    """
    try:
        mcp_get_api_params(context, {})
        
        # Validate prediction horizon
        try:
            horizon = PredictionHorizon(prediction_horizon)
        except ValueError:
            horizon = PredictionHorizon.MEDIUM_TERM
        
        analytics = PredictiveAnalytics()
        predictions = []
        
        for symbol in symbols:
            prediction = await analytics.predict_price_trend(symbol, horizon)
            if prediction and prediction.confidence >= confidence_threshold:
                predictions.append(asdict(prediction))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "prediction_horizon": horizon.value,
            "confidence_threshold": confidence_threshold,
            "symbols_analyzed": len(symbols),
            "predictions": predictions,
            "summary": {
                "total_predictions": len(predictions),
                "high_confidence_predictions": len([p for p in predictions if p["confidence"] > 0.8]),
                "bullish_predictions": len([p for p in predictions if p["trend_direction"] in ["bullish", "strong_bullish"]]),
                "bearish_predictions": len([p for p in predictions if p["trend_direction"] in ["bearish", "strong_bearish"]]),
                "average_confidence": f"{np.mean([p['confidence'] for p in predictions]):.1%}" if predictions else "0%"
            },
            "top_opportunities": predictions[:5] if predictions else [],
            "market_outlook": self._generate_market_outlook(predictions)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _generate_market_outlook(self, predictions: List[Dict]) -> str:
        """Generate overall market outlook based on predictions"""
        if not predictions:
            return "Insufficient data for market outlook"
        
        bullish_count = len([p for p in predictions if p["trend_direction"] in ["bullish", "strong_bullish"]])
        bearish_count = len([p for p in predictions if p["trend_direction"] in ["bearish", "strong_bearish"]])
        
        total_predictions = len(predictions)
        bullish_ratio = bullish_count / total_predictions
        bearish_ratio = bearish_count / total_predictions
        
        if bullish_ratio > 0.6:
            return "Bullish outlook - majority of predictions show upward trend"
        elif bearish_ratio > 0.6:
            return "Bearish outlook - majority of predictions show downward trend"
        else:
            return "Neutral outlook - mixed signals across predictions"


@mcp.tool(title="Forecast Volatility")
async def forecast_volatility(context: Context,
                            symbols: List[str] = Field(description="List of symbols to forecast volatility"),
                            forecast_horizon: str = Field("1w", description="Forecast horizon: 1d, 1w, 1m, 3m")):
    """
    Forecast volatility for specified symbols with trading implications
    
    Args:
        symbols: List of symbols to analyze
        forecast_horizon: Time horizon for volatility forecast
        
    Returns:
        Volatility forecasts with regime analysis and trading implications
    """
    try:
        mcp_get_api_params(context, {})
        
        # Validate forecast horizon
        try:
            horizon = PredictionHorizon(forecast_horizon)
        except ValueError:
            horizon = PredictionHorizon.MEDIUM_TERM
        
        analytics = PredictiveAnalytics()
        volatility_forecasts = []
        
        for symbol in symbols:
            forecast = await analytics.forecast_volatility(symbol, horizon)
            if forecast:
                volatility_forecasts.append(asdict(forecast))
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "forecast_horizon": horizon.value,
            "symbols_analyzed": len(symbols),
            "volatility_forecasts": volatility_forecasts,
            "summary": {
                "total_forecasts": len(volatility_forecasts),
                "low_volatility_symbols": len([f for f in volatility_forecasts if f["volatility_regime"] == "Low"]),
                "high_volatility_symbols": len([f for f in volatility_forecasts if f["volatility_regime"] == "High"]),
                "extreme_volatility_symbols": len([f for f in volatility_forecasts if f["volatility_regime"] == "Extreme"]),
                "average_forecasted_volatility": f"{np.mean([f['predicted_volatility'] for f in volatility_forecasts]):.1%}" if volatility_forecasts else "0%"
            },
            "volatility_regime_analysis": self._analyze_volatility_regimes(volatility_forecasts),
            "trading_strategies": self._get_volatility_trading_strategies(volatility_forecasts)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _analyze_volatility_regimes(self, forecasts: List[Dict]) -> Dict[str, Any]:
        """Analyze volatility regimes across symbols"""
        regimes = {}
        for forecast in forecasts:
            regime = forecast["volatility_regime"]
            if regime not in regimes:
                regimes[regime] = {"count": 0, "symbols": []}
            regimes[regime]["count"] += 1
            regimes[regime]["symbols"].append(forecast["symbol"])
        
        return {
            "regime_distribution": regimes,
            "market_volatility_sentiment": "Elevated" if regimes.get("High", {}).get("count", 0) > len(forecasts) * 0.4 else "Normal"
        }
    
    def _get_volatility_trading_strategies(self, forecasts: List[Dict]) -> List[str]:
        """Get trading strategies based on volatility forecasts"""
        strategies = []
        
        high_vol_symbols = [f for f in forecasts if f["volatility_regime"] == "High"]
        low_vol_symbols = [f for f in forecasts if f["volatility_regime"] == "Low"]
        
        if high_vol_symbols:
            strategies.append(f"Consider volatility selling strategies for {len(high_vol_symbols)} high-volatility symbols")
        
        if low_vol_symbols:
            strategies.append(f"Low volatility environment suitable for carry strategies in {len(low_vol_symbols)} symbols")
        
        strategies.extend([
            "Monitor VIX levels for market-wide volatility signals",
            "Consider dynamic hedging based on volatility forecasts",
            "Adjust position sizes based on forecasted volatility"
        ])
        
        return strategies


@mcp.tool(title="Analyze Asset Correlations")
async def analyze_asset_correlations(context: Context,
                                   symbols: List[str] = Field(description="List of symbols to analyze correlations"),
                                   correlation_threshold: float = Field(0.7, ge=0.5, le=0.95, description="Threshold for high correlation")):
    """
    Analyze correlations between asset pairs and provide diversification insights
    
    Args:
        symbols: List of symbols to analyze
        correlation_threshold: Threshold for considering high correlation
        
    Returns:
        Correlation analysis with diversification recommendations
    """
    try:
        mcp_get_api_params(context, {})
        
        analytics = PredictiveAnalytics()
        correlations = await analytics.analyze_correlations(symbols)
        
        # Filter high correlations
        high_correlations = [c for c in correlations if abs(c.current_correlation) > correlation_threshold]
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "symbols_analyzed": symbols,
            "correlation_analysis": [asdict(corr) for corr in correlations],
            "summary": {
                "total_pairs": len(correlations),
                "high_correlation_pairs": len(high_correlations),
                "average_correlation": f"{np.mean([c.current_correlation for c in correlations]):.2f}" if correlations else "0.00",
                "diversification_score": self._calculate_diversification_score(correlations)
            },
            "risk_alerts": self._generate_correlation_alerts(high_correlations),
            "rebalancing_recommendations": self._generate_correlation_recommendations(correlations)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _calculate_diversification_score(self, correlations: List[CorrelationAnalysis]) -> float:
        """Calculate portfolio diversification score"""
        if not correlations:
            return 0.0
        
        correlations_values = [abs(c.current_correlation) for c in correlations]
        avg_correlation = np.mean(correlations_values)
        
        # Lower average correlation = better diversification
        diversification_score = 1 - avg_correlation
        return max(0, min(1, diversification_score))
    
    def _generate_correlation_alerts(self, high_correlations: List[CorrelationAnalysis]) -> List[Dict]:
        """Generate alerts for high correlation pairs"""
        alerts = []
        
        for corr in high_correlations:
            alert = {
                "type": "High Correlation Alert",
                "assets": f"{corr.asset_pairs[0]} - {corr.asset_pairs[1]}",
                "correlation": f"{corr.current_correlation:.2f}",
                "severity": "High" if abs(corr.current_correlation) > 0.9 else "Medium",
                "message": f"High correlation detected between {corr.asset_pairs[0]} and {corr.asset_pairs[1]}"
            }
            alerts.append(alert)
        
        return alerts
    
    def _generate_correlation_recommendations(self, correlations: List[CorrelationAnalysis]) -> List[str]:
        """Generate rebalancing recommendations based on correlations"""
        recommendations = []
        
        # Find highly correlated pairs
        high_corr_pairs = [(c.asset_pairs[0], c.asset_pairs[1], c.current_correlation) 
                         for c in correlations if abs(c.current_correlation) > 0.8]
        
        if high_corr_pairs:
            recommendations.append("Consider reducing exposure to highly correlated assets")
            for asset1, asset2, corr in high_corr_pairs:
                recommendations.append(f"Monitor {asset1} and {asset2} correlation ({corr:.2f})")
        
        # Diversification suggestions
        low_corr_assets = [c.asset_pairs for c in correlations if abs(c.current_correlation) < 0.3]
        if low_corr_assets:
            recommendations.append("Assets with low correlation provide good diversification")
        
        recommendations.extend([
            "Regular correlation monitoring recommended",
            "Consider geographic and sector diversification",
            "Review correlation during market stress periods"
        ])
        
        return recommendations


@mcp.tool(title="Detect Market Anomalies")
async def detect_market_anomalies(context: Context,
                                symbols: List[str] = Field(description="List of symbols to monitor for anomalies"),
                                anomaly_types: Optional[List[str]] = Field(None, description="Specific anomaly types to detect")):
    """
    Detect market anomalies and unusual patterns across specified symbols
    
    Args:
        symbols: List of symbols to monitor
        anomaly_types: Specific types of anomalies to detect
        
    Returns:
        Detected anomalies with severity levels and recommended actions
    """
    try:
        mcp_get_api_params(context, {})
        
        analytics = PredictiveAnalytics()
        anomalies = await analytics.detect_anomalies(symbols)
        
        # Filter by anomaly types if specified
        if anomaly_types:
            try:
                type_enums = [AnomalyType(t) for t in anomaly_types]
                anomalies = [a for a in anomalies if a.anomaly_type in type_enums]
            except ValueError:
                logger.warning(f"Invalid anomaly types: {anomaly_types}")
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "symbols_monitored": symbols,
            "anomalies_detected": [asdict(anomaly) for anomaly in anomalies],
            "summary": {
                "total_anomalies": len(anomalies),
                "critical_anomalies": len([a for a in anomalies if a.severity == "Critical"]),
                "high_severity_anomalies": len([a for a in anomalies if a.severity == "High"]),
                "medium_severity_anomalies": len([a for a in anomalies if a.severity == "Medium"]),
                "anomaly_types": list(set([a.anomaly_type.value for a in anomalies]))
            },
            "priority_actions": self._generate_priority_actions(anomalies),
            "monitoring_recommendations": self._get_anomaly_monitoring_recommendations(anomalies)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _generate_priority_actions(self, anomalies: List[MarketAnomaly]) -> List[str]:
        """Generate priority actions based on detected anomalies"""
        actions = []
        
        critical_anomalies = [a for a in anomalies if a.severity == "Critical"]
        if critical_anomalies:
            actions.append("URGENT: Address critical anomalies immediately")
        
        volume_spikes = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME_SPIKE]
        if volume_spikes:
            actions.append("Monitor volume spike symbols for news and unusual activity")
        
        price_gaps = [a for a in anomalies if a.anomaly_type == AnomalyType.PRICE_GAP]
        if price_gaps:
            actions.append("Investigate price gap symbols for fundamental reasons")
        
        return actions
    
    def _get_anomaly_monitoring_recommendations(self, anomalies: List[MarketAnomaly]) -> List[str]:
        """Get monitoring recommendations based on detected anomalies"""
        recommendations = []
        
        if anomalies:
            recommendations.extend([
                "Increase monitoring frequency for anomalous symbols",
                "Check for relevant news and events",
                "Consider position adjustments if anomalies persist"
            ])
        else:
            recommendations.append("No anomalies detected - maintain standard monitoring")
        
        recommendations.extend([
            "Set up alerts for future anomaly detection",
            "Review historical anomaly patterns",
            "Maintain real-time monitoring capabilities"
        ])
        
        return recommendations


@mcp.tool(title="Get Economic Indicators Analysis")
async def get_economic_indicators_analysis(context: Context,
                                         indicators: Optional[List[str]] = Field(None, description="Specific indicators to analyze"),
                                         impact_threshold: float = Field(0.02, ge=0.01, le=0.10, description="Minimum change threshold for impact assessment")):
    """
    Analyze economic indicators and their potential impact on markets
    
    Args:
        indicators: Specific economic indicators to analyze
        impact_threshold: Minimum percentage change for impact assessment
        
    Returns:
        Economic indicators analysis with market impact assessment
    """
    try:
        mcp_get_api_params(context, {})
        
        analytics = PredictiveAnalytics()
        economic_indicators = await analytics.get_economic_indicators()
        
        # Filter by specified indicators if any
        if indicators:
            economic_indicators = [i for i in economic_indicators 
                                 if i.indicator_name.lower() in [ind.lower() for ind in indicators]]
        
        # Classify by impact
        high_impact = [i for i in economic_indicators if abs(i.change_percentage) > impact_threshold * 100]
        
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "economic_indicators": [asdict(indicator) for indicator in economic_indicators],
            "summary": {
                "total_indicators": len(economic_indicators),
                "high_impact_indicators": len(high_impact),
                "positive_changes": len([i for i in economic_indicators if i.change_percentage > 0]),
                "negative_changes": len([i for i in economic_indicators if i.change_percentage < 0]),
                "average_change": f"{np.mean([i.change_percentage for i in economic_indicators]):.1f}%" if economic_indicators else "0.0%"
            },
            "market_impact_assessment": self._assess_market_impact(economic_indicators),
            "trading_implications": self._get_economic_trading_implications(economic_indicators),
            "monitoring_schedule": self._create_economic_monitoring_schedule(economic_indicators)
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _assess_market_impact(self, indicators: List[EconomicIndicator]) -> Dict[str, str]:
        """Assess overall market impact from economic indicators"""
        high_impact_indicators = [i for i in indicators if i.impact_on_market == "High Impact"]
        
        if len(high_impact_indicators) > 3:
            overall_impact = "Significant market impact expected"
        elif len(high_impact_indicators) > 1:
            overall_impact = "Moderate market impact expected"
        else:
            overall_impact = "Limited market impact expected"
        
        return {
            "overall_impact": overall_impact,
            "key_factors": [i.indicator_name for i in high_impact_indicators[:3]],
            "confidence_level": "High" if len(high_impact_indicators) > 2 else "Medium"
        }
    
    def _get_economic_trading_implications(self, indicators: List[EconomicIndicator]) -> List[str]:
        """Get trading implications based on economic indicators"""
        implications = []
        
        # Federal Funds Rate
        fed_rate = next((i for i in indicators if "federal funds rate" in i.indicator_name.lower()), None)
        if fed_rate and fed_rate.change_percentage > 0:
            implications.append("Rising rates may pressure growth stocks and increase bond yields")
        
        # Inflation
        inflation = next((i for i in indicators if "inflation" in i.indicator_name.lower()), None)
        if inflation and inflation.change_percentage > 0:
            implications.append("Rising inflation may benefit inflation-protected assets")
        
        # Unemployment
        unemployment = next((i for i in indicators if "unemployment" in i.indicator_name.lower()), None)
        if unemployment and unemployment.change_percentage < 0:
            implications.append("Low unemployment supports consumer spending and cyclical stocks")
        
        # General implications
        implications.extend([
            "Monitor central bank communications for policy signals",
            "Consider sector rotation based on economic trends",
            "Maintain awareness of geopolitical developments"
        ])
        
        return implications
    
    def _create_economic_monitoring_schedule(self, indicators: List[EconomicIndicator]) -> Dict[str, List[str]]:
        """Create monitoring schedule for economic indicators"""
        schedule = {
            "daily": ["VIX", "Market sentiment indicators"],
            "weekly": ["Federal Funds Rate", "Unemployment Rate"],
            "monthly": ["Inflation Rate", "GDP Growth"],
            "quarterly": ["Economic forecasts", "Central bank meetings"]
        }
        
        # Customize based on current indicators
        relevant_indicators = [i.indicator_name.lower() for i in indicators]
        
        custom_schedule = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        for indicator in indicators:
            if indicator.impact_on_market == "High Impact":
                custom_schedule["high_priority"].append(indicator.indicator_name)
            elif indicator.impact_on_market == "Medium Impact":
                custom_schedule["medium_priority"].append(indicator.indicator_name)
            else:
                custom_schedule["low_priority"].append(indicator.indicator_name)
        
        return custom_schedule
