"""
Alert System for OrPaynter Financial MCP Server

Provides comprehensive alert capabilities including:
- Price threshold alerts for all monitored assets
- News sentiment analysis integration
- Anomaly detection for unusual market activity
- Portfolio performance alerts
- Custom alert rules and notification management
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict
import uuid
import schedule
import time
from threading import Thread

from fastmcp import FastMCP, Context
from pydantic import Field
from loguru import logger

from finance_trading_ai_agents_mcp.utils.common_utils import mcp_get_api_params, show_mcp_result
from finance_trading_ai_agents_mcp.mcp_services.global_instance import McpGlobalVar
from finance_trading_ai_agents_mcp.mcp_result_control.common_control import CommonControl
from aitrados_api.trade_middleware_service.trade_middleware_service_instance import AitradosApiServiceInstance
from aitrados_api.common_lib.contant import ApiDataFormat


class AlertType(Enum):
    PRICE_THRESHOLD = "price_threshold"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY = "volatility"
    NEWS_SENTIMENT = "news_sentiment"
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    CORRELATION_BREAK = "correlation_break"
    TECHNICAL_SIGNAL = "technical_signal"
    ECONOMIC_EVENT = "economic_event"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WEB_UI = "web_ui"
    API = "api"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    alert_type: AlertType
    symbol: str
    parameters: Dict[str, Any]
    severity: AlertSeverity
    notification_channels: List[NotificationChannel]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    cooldown_period: int = 300  # 5 minutes in seconds
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    symbol: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    data: Dict[str, Any]
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class PriceThreshold:
    """Price threshold configuration"""
    symbol: str
    upper_threshold: Optional[float] = None
    lower_threshold: Optional[float] = None
    percentage_change: Optional[float] = None  # e.g., 5% change triggers alert
    volume_threshold: Optional[float] = None
    volatility_threshold: Optional[float] = None


@dataclass
class NewsAlertConfig:
    """News sentiment alert configuration"""
    symbols: List[str]
    sentiment_threshold: float = 0.3  # Sentiment score threshold
    min_articles: int = 1  # Minimum number of articles
    time_window: int = 3600  # Time window in seconds (1 hour)
    sources: Optional[List[str]] = None


class AlertManager:
    """Core alert management system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Alert monitoring state
        self.price_data_cache: Dict[str, Dict] = {}
        self.news_cache: Dict[str, List] = {}
        self.last_price_checks: Dict[str, datetime] = {}
        self.last_news_checks: Dict[str, datetime] = {}
        
        # Performance tracking
        self.alert_stats = {
            "total_alerts": 0,
            "alerts_by_type": defaultdict(int),
            "alerts_by_severity": defaultdict(int),
            "response_times": [],
            "false_positives": 0
        }
        
        # Start background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
    
    def register_notification_handler(self, channel: NotificationChannel, handler: Callable):
        """Register notification handler for a specific channel"""
        self.notification_handlers[channel] = handler
        logger.info(f"Registered notification handler for {channel.value}")
    
    def create_alert_rule(self, rule: AlertRule) -> bool:
        """Create a new alert rule"""
        try:
            if rule.rule_id in self.alert_rules:
                logger.warning(f"Alert rule {rule.rule_id} already exists")
                return False
            
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Created alert rule: {rule.name} for {rule.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {e}")
            return False
    
    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule"""
        try:
            if rule_id not in self.alert_rules:
                logger.warning(f"Alert rule {rule_id} not found")
                return False
            
            rule = self.alert_rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            logger.info(f"Updated alert rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert rule: {e}")
            return False
    
    def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule"""
        try:
            if rule_id in self.alert_rules:
                rule = self.alert_rules.pop(rule_id)
                logger.info(f"Deleted alert rule: {rule.name}")
                return True
            else:
                logger.warning(f"Alert rule {rule_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting alert rule: {e}")
            return False
    
    async def check_price_alerts(self, symbol: str) -> List[Alert]:
        """Check price-based alerts for a symbol"""
        try:
            alerts = []
            
            # Get current price data
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return alerts
            
            # Cache price data
            self.price_data_cache[symbol] = {
                "price": current_price,
                "timestamp": datetime.now()
            }
            
            # Check all applicable rules for this symbol
            for rule in self.alert_rules.values():
                if (rule.symbol == symbol and 
                    rule.is_active and 
                    rule.alert_type == AlertType.PRICE_THRESHOLD):
                    
                    alert = await self._check_price_threshold_rule(rule, current_price)
                    if alert:
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking price alerts for {symbol}: {e}")
            return []
    
    async def check_volume_alerts(self, symbol: str) -> List[Alert]:
        """Check volume-based alerts for a symbol"""
        try:
            alerts = []
            
            # Get volume data
            volume_data = await self._get_volume_data(symbol)
            if not volume_data:
                return alerts
            
            # Check volume spike rules
            for rule in self.alert_rules.values():
                if (rule.symbol == symbol and 
                    rule.is_active and 
                    rule.alert_type == AlertType.VOLUME_SPIKE):
                    
                    alert = await self._check_volume_spike_rule(rule, volume_data)
                    if alert:
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking volume alerts for {symbol}: {e}")
            return []
    
    async def check_news_sentiment_alerts(self) -> List[Alert]:
        """Check news sentiment alerts across all symbols"""
        try:
            alerts = []
            
            # Get symbols with news sentiment rules
            news_symbols = set()
            for rule in self.alert_rules.values():
                if (rule.is_active and 
                    rule.alert_type == AlertType.NEWS_SENTIMENT and
                    "symbols" in rule.parameters):
                    news_symbols.update(rule.parameters["symbols"])
            
            # Check news for each symbol
            for symbol in news_symbols:
                news_data = await self._get_news_data(symbol)
                if not news_data:
                    continue
                
                # Calculate sentiment
                sentiment_score = self._calculate_news_sentiment(news_data)
                
                # Check sentiment rules
                for rule in self.alert_rules.values():
                    if (rule.symbol == symbol and 
                        rule.is_active and 
                        rule.alert_type == AlertType.NEWS_SENTIMENT):
                        
                        alert = await self._check_sentiment_rule(rule, sentiment_score, news_data)
                        if alert:
                            alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking news sentiment alerts: {e}")
            return []
    
    async def check_portfolio_alerts(self, portfolio_data: Dict[str, Any]) -> List[Alert]:
        """Check portfolio performance alerts"""
        try:
            alerts = []
            
            for rule in self.alert_rules.values():
                if (rule.is_active and 
                    rule.alert_type == AlertType.PORTFOLIO_PERFORMANCE):
                    
                    alert = await self._check_portfolio_rule(rule, portfolio_data)
                    if alert:
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking portfolio alerts: {e}")
            return []
    
    async def trigger_alert(self, alert: Alert) -> bool:
        """Trigger an alert and send notifications"""
        try:
            # Check cooldown period
            rule = self.alert_rules.get(alert.rule_id)
            if rule and rule.last_triggered:
                time_since_last = (datetime.now() - rule.last_triggered).total_seconds()
                if time_since_last < rule.cooldown_period:
                    logger.info(f"Alert {alert.alert_id} skipped due to cooldown period")
                    return False
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update rule last triggered time
            if rule:
                rule.last_triggered = datetime.now()
            
            # Update statistics
            self.alert_stats["total_alerts"] += 1
            self.alert_stats["alerts_by_type"][alert.alert_type.value] += 1
            self.alert_stats["alerts_by_severity"][alert.severity.value] += 1
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.info(f"Alert triggered: {alert.message}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert.alert_id}: {e}")
            return False
    
    async def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            else:
                logger.warning(f"Alert {alert_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                # Move to history
                resolved_alert = self.active_alerts.pop(alert_id)
                
                logger.info(f"Alert {alert_id} resolved: {resolution_note}")
                return True
            else:
                logger.warning(f"Alert {alert_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def start_monitoring(self):
        """Start background alert monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background alert monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Schedule price checks
                schedule.every(1).minutes.do(self._check_all_price_alerts)
                schedule.every(5).minutes.do(self._check_all_news_alerts)
                schedule.every(15).minutes.do(self._check_all_volume_alerts)
                
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            params = {
                "full_symbol": f"STOCK:US:{symbol}",
                "interval": "DAY",
                "limit": 1,
                "format": ApiDataFormat.JSON
            }
            
            ohlc_data = await AitradosApiServiceInstance.api_client.ohlc.a_ohlcs_latest(**params)
            
            if ohlc_data and len(ohlc_data) > 0:
                return float(ohlc_data[0].get('close', 0))
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _get_volume_data(self, symbol: str) -> Optional[Dict]:
        """Get volume data for a symbol"""
        try:
            params = {
                "full_symbol": f"STOCK:US:{symbol}",
                "interval": "DAY",
                "limit": 10,
                "format": ApiDataFormat.JSON
            }
            
            ohlc_data = await AitradosApiServiceInstance.api_client.ohlc.a_ohlcs_latest(**params)
            
            if ohlc_data:
                volumes = [float(data.get('volume', 0)) for data in ohlc_data]
                return {
                    "current_volume": volumes[-1] if volumes else 0,
                    "average_volume": np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1] if volumes else 0,
                    "volumes": volumes
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting volume data for {symbol}: {e}")
            return None
    
    async def _get_news_data(self, symbol: str) -> List[Dict]:
        """Get news data for a symbol"""
        try:
            params = {
                "full_symbol": f"STOCK:US:{symbol}",
                "limit": 20
            }
            
            news_data = await AitradosApiServiceInstance.api_client.news.a_news_latest(**params)
            
            return news_data if news_data else []
            
        except Exception as e:
            logger.error(f"Error getting news data for {symbol}: {e}")
            return []
    
    def _calculate_news_sentiment(self, news_data: List[Dict]) -> float:
        """Calculate sentiment score from news data"""
        if not news_data:
            return 0.0
        
        # Simple sentiment calculation (in production, would use NLP models)
        positive_keywords = ['profit', 'growth', 'gain', 'rise', 'increase', 'positive', 'strong']
        negative_keywords = ['loss', 'decline', 'fall', 'decrease', 'negative', 'weak', 'concern']
        
        total_sentiment = 0
        processed_articles = 0
        
        for article in news_data:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            text = f"{title} {content}"
            
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            if positive_count + negative_count > 0:
                article_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                total_sentiment += article_sentiment
                processed_articles += 1
        
        return total_sentiment / processed_articles if processed_articles > 0 else 0.0
    
    async def _check_price_threshold_rule(self, rule: AlertRule, current_price: float) -> Optional[Alert]:
        """Check price threshold rule"""
        try:
            parameters = rule.parameters
            alert_triggered = False
            alert_message = ""
            alert_data = {"current_price": current_price}
            
            # Check absolute thresholds
            if "upper_threshold" in parameters and current_price > parameters["upper_threshold"]:
                alert_triggered = True
                alert_message = f"{rule.symbol} price ${current_price:.2f} exceeded upper threshold ${parameters['upper_threshold']:.2f}"
                alert_data["upper_threshold"] = parameters["upper_threshold"]
            
            elif "lower_threshold" in parameters and current_price < parameters["lower_threshold"]:
                alert_triggered = True
                alert_message = f"{rule.symbol} price ${current_price:.2f} fell below lower threshold ${parameters['lower_threshold']:.2f}"
                alert_data["lower_threshold"] = parameters["lower_threshold"]
            
            # Check percentage change
            if "percentage_change" in parameters and rule.symbol in self.price_data_cache:
                previous_price = self.price_data_cache[rule.symbol].get("price")
                if previous_price:
                    change_pct = abs((current_price - previous_price) / previous_price) * 100
                    if change_pct >= parameters["percentage_change"]:
                        alert_triggered = True
                        direction = "increased" if current_price > previous_price else "decreased"
                        alert_message = f"{rule.symbol} price {direction} by {change_pct:.1f}%"
                        alert_data["percentage_change"] = change_pct
                        alert_data["previous_price"] = previous_price
            
            if alert_triggered:
                return Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    alert_type=AlertType.PRICE_THRESHOLD,
                    symbol=rule.symbol,
                    message=alert_message,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    triggered_at=datetime.now(),
                    data=alert_data
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking price threshold rule: {e}")
            return None
    
    async def _check_volume_spike_rule(self, rule: AlertRule, volume_data: Dict) -> Optional[Alert]:
        """Check volume spike rule"""
        try:
            current_volume = volume_data["current_volume"]
            average_volume = volume_data["average_volume"]
            parameters = rule.parameters
            
            spike_multiplier = parameters.get("spike_multiplier", 2.0)
            
            if current_volume > average_volume * spike_multiplier:
                spike_ratio = current_volume / average_volume if average_volume > 0 else 0
                
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    alert_type=AlertType.VOLUME_SPIKE,
                    symbol=rule.symbol,
                    message=f"{rule.symbol} volume spike: {spike_ratio:.1f}x average",
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    triggered_at=datetime.now(),
                    data={
                        "current_volume": current_volume,
                        "average_volume": average_volume,
                        "spike_ratio": spike_ratio
                    }
                )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking volume spike rule: {e}")
            return None
    
    async def _check_sentiment_rule(self, rule: AlertRule, sentiment_score: float, news_data: List[Dict]) -> Optional[Alert]:
        """Check news sentiment rule"""
        try:
            parameters = rule.parameters
            threshold = parameters.get("sentiment_threshold", 0.3)
            min_articles = parameters.get("min_articles", 1)
            
            if len(news_data) >= min_articles and abs(sentiment_score) >= threshold:
                sentiment_type = "positive" if sentiment_score > 0 else "negative"
                
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    alert_type=AlertType.NEWS_SENTIMENT,
                    symbol=rule.symbol,
                    message=f"{rule.symbol} {sentiment_type} news sentiment detected (score: {sentiment_score:.2f})",
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    triggered_at=datetime.now(),
                    data={
                        "sentiment_score": sentiment_score,
                        "article_count": len(news_data),
                        "sentiment_type": sentiment_type
                    }
                )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking sentiment rule: {e}")
            return None
    
    async def _check_portfolio_rule(self, rule: AlertRule, portfolio_data: Dict[str, Any]) -> Optional[Alert]:
        """Check portfolio performance rule"""
        try:
            parameters = rule.parameters
            positions = portfolio_data.get("positions", [])
            
            if not positions:
                return None
            
            portfolio_value = sum(pos.get("market_value", 0) for pos in positions)
            alert_triggered = False
            alert_message = ""
            alert_data = {"portfolio_value": portfolio_value}
            
            # Check total portfolio value thresholds
            if "min_value" in parameters and portfolio_value < parameters["min_value"]:
                alert_triggered = True
                alert_message = f"Portfolio value ${portfolio_value:.2f} below minimum ${parameters['min_value']:.2f}"
                alert_data["min_value"] = parameters["min_value"]
            
            elif "max_value" in parameters and portfolio_value > parameters["max_value"]:
                alert_triggered = True
                alert_message = f"Portfolio value ${portfolio_value:.2f} above maximum ${parameters['max_value']:.2f}"
                alert_data["max_value"] = parameters["max_value"]
            
            # Check concentration risk
            if "max_concentration" in parameters:
                max_concentration = 0
                for pos in positions:
                    weight = pos.get("market_value", 0) / portfolio_value if portfolio_value > 0 else 0
                    max_concentration = max(max_concentration, weight)
                
                if max_concentration > parameters["max_concentration"]:
                    alert_triggered = True
                    alert_message = f"Portfolio concentration risk: {max_concentration:.1%} exceeds threshold {parameters['max_concentration']:.1%}"
                    alert_data["max_concentration"] = max_concentration
                    alert_data["threshold"] = parameters["max_concentration"]
            
            if alert_triggered:
                return Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    alert_type=AlertType.PORTFOLIO_PERFORMANCE,
                    symbol="PORTFOLIO",
                    message=alert_message,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    triggered_at=datetime.now(),
                    data=alert_data
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking portfolio rule: {e}")
            return None
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        try:
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                return
            
            notification_data = {
                "alert_id": alert.alert_id,
                "message": alert.message,
                "severity": alert.severity.value,
                "symbol": alert.symbol,
                "alert_type": alert.alert_type.value,
                "triggered_at": alert.triggered_at.isoformat(),
                "data": alert.data
            }
            
            # Send through each configured channel
            for channel in rule.notification_channels:
                if channel in self.notification_handlers:
                    try:
                        await self.notification_handlers[channel](notification_data)
                    except Exception as e:
                        logger.error(f"Error sending notification via {channel.value}: {e}")
                else:
                    logger.warning(f"No handler registered for notification channel: {channel.value}")
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def _check_all_price_alerts(self):
        """Background task to check all price alerts"""
        try:
            symbols = set(rule.symbol for rule in self.alert_rules.values() 
                         if rule.alert_type == AlertType.PRICE_THRESHOLD and rule.is_active)
            
            for symbol in symbols:
                asyncio.create_task(self.check_price_alerts(symbol))
                
        except Exception as e:
            logger.error(f"Error in background price alert check: {e}")
    
    def _check_all_news_alerts(self):
        """Background task to check all news alerts"""
        try:
            asyncio.create_task(self.check_news_sentiment_alerts())
        except Exception as e:
            logger.error(f"Error in background news alert check: {e}")
    
    def _check_all_volume_alerts(self):
        """Background task to check all volume alerts"""
        try:
            symbols = set(rule.symbol for rule in self.alert_rules.values() 
                         if rule.alert_type == AlertType.VOLUME_SPIKE and rule.is_active)
            
            for symbol in symbols:
                asyncio.create_task(self.check_volume_alerts(symbol))
                
        except Exception as e:
            logger.error(f"Error in background volume alert check: {e}")


# Global alert manager instance
alert_manager = AlertManager()


# FastMCP instance for Alert System
mcp = FastMCP("orpaynter_alert_system")
mcp_app = mcp.http_app(path="/", transport="streamable-http", stateless_http=True)


@mcp.tool(title="Create Alert Rule")
async def create_alert_rule(context: Context,
                          rule_name: str = Field(description="Name of the alert rule"),
                          alert_type: str = Field(description="Type of alert: price_threshold, volume_spike, news_sentiment, portfolio_performance"),
                          symbol: str = Field(description="Symbol to monitor"),
                          parameters: Dict[str, Any] = Field(description="Alert parameters specific to the type"),
                          severity: str = Field("medium", description="Alert severity: low, medium, high, critical"),
                          notification_channels: List[str] = Field(["web_ui"], description="Notification channels"),
                          cooldown_period: int = Field(300, description="Cooldown period in seconds")):
    """
    Create a new alert rule for monitoring financial instruments
    
    Args:
        rule_name: Human-readable name for the rule
        alert_type: Type of alert to create
        symbol: Symbol to monitor
        parameters: Alert-specific parameters
        severity: Alert severity level
        notification_channels: How to send notifications
        cooldown_period: Minimum time between alerts
        
    Returns:
        Created alert rule details
    """
    try:
        mcp_get_api_params(context, {})
        
        # Validate inputs
        try:
            alert_type_enum = AlertType(alert_type)
            severity_enum = AlertSeverity(severity)
            channels_enum = [NotificationChannel(ch) for ch in notification_channels]
        except ValueError as e:
            raise ValueError(f"Invalid parameter value: {e}")
        
        # Create alert rule
        rule_id = str(uuid.uuid4())
        rule = AlertRule(
            rule_id=rule_id,
            name=rule_name,
            alert_type=alert_type_enum,
            symbol=symbol,
            parameters=parameters,
            severity=severity_enum,
            notification_channels=channels_enum,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            cooldown_period=cooldown_period
        )
        
        success = alert_manager.create_alert_rule(rule)
        
        if success:
            result = {
                "rule_created": True,
                "rule_id": rule_id,
                "rule_details": asdict(rule),
                "message": f"Alert rule '{rule_name}' created successfully"
            }
        else:
            result = {
                "rule_created": False,
                "error": "Failed to create alert rule"
            }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="List Alert Rules")
async def list_alert_rules(context: Context,
                         alert_type: Optional[str] = Field(None, description="Filter by alert type"),
                         symbol: Optional[str] = Field(None, description="Filter by symbol"),
                         is_active: Optional[bool] = Field(None, description="Filter by active status")):
    """
    List all alert rules with optional filtering
    
    Args:
        alert_type: Filter by alert type
        symbol: Filter by symbol
        is_active: Filter by active status
        
    Returns:
        List of alert rules matching criteria
    """
    try:
        mcp_get_api_params(context, {})
        
        # Filter rules
        filtered_rules = list(alert_manager.alert_rules.values())
        
        if alert_type:
            try:
                alert_type_enum = AlertType(alert_type)
                filtered_rules = [r for r in filtered_rules if r.alert_type == alert_type_enum]
            except ValueError:
                pass  # Invalid type, no filtering
        
        if symbol:
            filtered_rules = [r for r in filtered_rules if r.symbol == symbol]
        
        if is_active is not None:
            filtered_rules = [r for r in filtered_rules if r.is_active == is_active]
        
        result = {
            "total_rules": len(filtered_rules),
            "alert_rules": [asdict(rule) for rule in filtered_rules],
            "filters_applied": {
                "alert_type": alert_type,
                "symbol": symbol,
                "is_active": is_active
            },
            "summary": {
                "active_rules": len([r for r in filtered_rules if r.is_active]),
                "inactive_rules": len([r for r in filtered_rules if not r.is_active]),
                "rules_by_type": {t.value: len([r for r in filtered_rules if r.alert_type == t]) 
                                for t in AlertType},
                "rules_by_severity": {s.value: len([r for r in filtered_rules if r.severity == s]) 
                                    for s in AlertSeverity}
            }
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Update Alert Rule")
async def update_alert_rule(context: Context,
                          rule_id: str = Field(description="ID of the rule to update"),
                          updates: Dict[str, Any] = Field(description="Fields to update")):
    """
    Update an existing alert rule
    
    Args:
        rule_id: ID of the rule to update
        updates: Dictionary of fields to update
        
    Returns:
        Update result
    """
    try:
        mcp_get_api_params(context, {})
        
        success = alert_manager.update_alert_rule(rule_id, updates)
        
        if success:
            updated_rule = alert_manager.alert_rules.get(rule_id)
            result = {
                "rule_updated": True,
                "rule_id": rule_id,
                "updated_rule": asdict(updated_rule) if updated_rule else None,
                "message": "Alert rule updated successfully"
            }
        else:
            result = {
                "rule_updated": False,
                "error": "Failed to update alert rule or rule not found"
            }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Delete Alert Rule")
async def delete_alert_rule(context: Context,
                          rule_id: str = Field(description="ID of the rule to delete")):
    """
    Delete an alert rule
    
    Args:
        rule_id: ID of the rule to delete
        
    Returns:
        Deletion result
    """
    try:
        mcp_get_api_params(context, {})
        
        success = alert_manager.delete_alert_rule(rule_id)
        
        result = {
            "rule_deleted": success,
            "rule_id": rule_id,
            "message": "Alert rule deleted successfully" if success else "Alert rule not found"
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Get Active Alerts")
async def get_active_alerts(context: Context,
                          severity: Optional[str] = Field(None, description="Filter by severity"),
                          alert_type: Optional[str] = Field(None, description="Filter by alert type"),
                          symbol: Optional[str] = Field(None, description="Filter by symbol"),
                          status: Optional[str] = Field(None, description="Filter by status")):
    """
    Get current active alerts with optional filtering
    
    Args:
        severity: Filter by severity level
        alert_type: Filter by alert type
        symbol: Filter by symbol
        status: Filter by alert status
        
    Returns:
        List of active alerts
    """
    try:
        mcp_get_api_params(context, {})
        
        # Filter alerts
        filtered_alerts = list(alert_manager.active_alerts.values())
        
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity_enum]
            except ValueError:
                pass
        
        if alert_type:
            try:
                alert_type_enum = AlertType(alert_type)
                filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type_enum]
            except ValueError:
                pass
        
        if symbol:
            filtered_alerts = [a for a in filtered_alerts if a.symbol == symbol]
        
        if status:
            try:
                status_enum = AlertStatus(status)
                filtered_alerts = [a for a in filtered_alerts if a.status == status_enum]
            except ValueError:
                pass
        
        result = {
            "total_active_alerts": len(filtered_alerts),
            "alerts": [asdict(alert) for alert in filtered_alerts],
            "filters_applied": {
                "severity": severity,
                "alert_type": alert_type,
                "symbol": symbol,
                "status": status
            },
            "summary": {
                "alerts_by_severity": {s.value: len([a for a in filtered_alerts if a.severity == s]) 
                                     for s in AlertSeverity},
                "alerts_by_type": {t.value: len([a for a in filtered_alerts if a.alert_type == t]) 
                                 for t in AlertType},
                "oldest_alert_age": self._calculate_oldest_alert_age(filtered_alerts) if filtered_alerts else None
            }
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _calculate_oldest_alert_age(self, alerts: List[Alert]) -> str:
        """Calculate age of oldest alert"""
        if not alerts:
            return "0 minutes"
        
        oldest_alert = min(alerts, key=lambda a: a.triggered_at)
        age = datetime.now() - oldest_alert.triggered_at
        
        if age.days > 0:
            return f"{age.days} days"
        elif age.seconds > 3600:
            return f"{age.seconds // 3600} hours"
        else:
            return f"{age.seconds // 60} minutes"


@mcp.tool(title="Acknowledge Alert")
async def acknowledge_alert(context: Context,
                          alert_id: str = Field(description="ID of the alert to acknowledge"),
                          user: str = Field("system", description="User acknowledging the alert")):
    """
    Acknowledge an active alert
    
    Args:
        alert_id: ID of the alert to acknowledge
        user: User acknowledging the alert
        
    Returns:
        Acknowledgment result
    """
    try:
        mcp_get_api_params(context, {})
        
        success = await alert_manager.acknowledge_alert(alert_id, user)
        
        result = {
            "alert_acknowledged": success,
            "alert_id": alert_id,
            "acknowledged_by": user,
            "acknowledged_at": datetime.now().isoformat(),
            "message": "Alert acknowledged successfully" if success else "Alert not found or already resolved"
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Resolve Alert")
async def resolve_alert(context: Context,
                      alert_id: str = Field(description="ID of the alert to resolve"),
                      resolution_note: str = Field("", description="Note explaining the resolution")):
    """
    Resolve an active alert
    
    Args:
        alert_id: ID of the alert to resolve
        resolution_note: Note explaining how the alert was resolved
        
    Returns:
        Resolution result
    """
    try:
        mcp_get_api_params(context, {})
        
        success = await alert_manager.resolve_alert(alert_id, resolution_note)
        
        result = {
            "alert_resolved": success,
            "alert_id": alert_id,
            "resolution_note": resolution_note,
            "resolved_at": datetime.now().isoformat(),
            "message": "Alert resolved successfully" if success else "Alert not found"
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Get Alert Statistics")
async def get_alert_statistics(context: Context,
                             time_period: str = Field("24h", description="Time period for statistics: 1h, 24h, 7d, 30d")):
    """
    Get alert system statistics and performance metrics
    
    Args:
        time_period: Time period for statistics
        
    Returns:
        Alert system statistics and performance metrics
    """
    try:
        mcp_get_api_params(context, {})
        
        # Calculate time period
        now = datetime.now()
        if time_period == "1h":
            cutoff_time = now - timedelta(hours=1)
        elif time_period == "24h":
            cutoff_time = now - timedelta(days=1)
        elif time_period == "7d":
            cutoff_time = now - timedelta(days=7)
        elif time_period == "30d":
            cutoff_time = now - timedelta(days=30)
        else:
            cutoff_time = now - timedelta(days=1)  # Default to 24h
        
        # Filter alerts by time period
        period_alerts = [a for a in alert_manager.alert_history if a.triggered_at >= cutoff_time]
        
        result = {
            "time_period": time_period,
            "statistics": {
                "total_alerts": len(period_alerts),
                "active_alerts": len(alert_manager.active_alerts),
                "alerts_by_type": dict(alert_manager.alert_stats["alerts_by_type"]),
                "alerts_by_severity": dict(alert_manager.alert_stats["alerts_by_severity"]),
                "average_response_time": f"{np.mean(alert_manager.alert_stats['response_times']):.1f} minutes" if alert_manager.alert_stats['response_times'] else "N/A",
                "false_positive_rate": f"{alert_manager.alert_stats['false_positives'] / max(1, alert_manager.alert_stats['total_alerts']) * 100:.1f}%"
            },
            "rule_statistics": {
                "total_rules": len(alert_manager.alert_rules),
                "active_rules": len([r for r in alert_manager.alert_rules.values() if r.is_active]),
                "most_used_alert_type": max(alert_manager.alert_stats["alerts_by_type"].items(), key=lambda x: x[1])[0] if alert_manager.alert_stats["alerts_by_type"] else "None",
                "rules_by_severity": {s.value: len([r for r in alert_manager.alert_rules.values() if r.severity == s]) for s in AlertSeverity}
            },
            "performance_metrics": {
                "monitoring_uptime": "Active" if alert_manager.monitoring_active else "Inactive",
                "last_alert_time": max([a.triggered_at for a in period_alerts]).isoformat() if period_alerts else "None",
                "alert_frequency": f"{len(period_alerts) / max(1, (now - cutoff_time).total_seconds() / 3600):.1f} alerts/hour"
            },
            "recommendations": self._generate_alert_recommendations(alert_manager.alert_stats, len(alert_manager.active_alerts))
        }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
    
    def _generate_alert_recommendations(self, stats: Dict, active_alerts: int) -> List[str]:
        """Generate recommendations based on alert statistics"""
        recommendations = []
        
        # High alert volume
        if stats["total_alerts"] > 50:
            recommendations.append("High alert volume detected - consider adjusting thresholds")
        
        # Many false positives
        false_positive_rate = stats["false_positives"] / max(1, stats["total_alerts"])
        if false_positive_rate > 0.3:
            recommendations.append("High false positive rate - review alert rules")
        
        # Active alerts
        if active_alerts > 10:
            recommendations.append("Many active alerts - consider prioritizing critical ones")
        
        # General recommendations
        recommendations.extend([
            "Regularly review and update alert thresholds",
            "Monitor alert performance and adjust as needed",
            "Consider correlation between different alert types",
            "Implement alert fatigue management strategies"
        ])
        
        return recommendations


@mcp.tool(title="Start Alert Monitoring")
async def start_alert_monitoring(context: Context):
    """
    Start the background alert monitoring system
    
    Returns:
        Monitoring start confirmation
    """
    try:
        mcp_get_api_params(context, {})
        
        if not alert_manager.monitoring_active:
            alert_manager.start_monitoring()
            result = {
                "monitoring_started": True,
                "status": "Background monitoring is now active",
                "message": "Alert monitoring system started successfully"
            }
        else:
            result = {
                "monitoring_started": False,
                "status": "Background monitoring was already active",
                "message": "Alert monitoring system is already running"
            }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)


@mcp.tool(title="Stop Alert Monitoring")
async def stop_alert_monitoring(context: Context):
    """
    Stop the background alert monitoring system
    
    Returns:
        Monitoring stop confirmation
    """
    try:
        mcp_get_api_params(context, {})
        
        if alert_manager.monitoring_active:
            alert_manager.stop_monitoring()
            result = {
                "monitoring_stopped": True,
                "status": "Background monitoring has been stopped",
                "message": "Alert monitoring system stopped successfully"
            }
        else:
            result = {
                "monitoring_stopped": False,
                "status": "Background monitoring was not active",
                "message": "Alert monitoring system is not running"
            }
        
        show_mcp_result(mcp, json.dumps(result, indent=2))
        return json.dumps(result)
        
    except Exception as e:
        error_result = {"error": str(e), "timestamp": datetime.now().isoformat()}
        show_mcp_result(mcp, json.dumps(error_result), True)
        return json.dumps(error_result)
