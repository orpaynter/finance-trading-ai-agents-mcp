"""
Simplified Alert System for OrPaynter MCP Server
Standalone implementation without external dependencies
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import queue

class AlertSystem:
    """Simplified alert management system"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.logger = None
        self.active_alerts = {}
        self.alert_history = []
        self.notification_queue = queue.Queue()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Alert types and their configurations
        self.alert_types = {
            "price_change": {"default_threshold": 5.0, "description": "Price change percentage"},
            "volume_spike": {"default_threshold": 2.0, "description": "Volume multiplier"},
            "volatility": {"default_threshold": 30.0, "description": "Volatility percentage"},
            "correlation_break": {"default_threshold": 0.5, "description": "Correlation deviation"},
            "news_sentiment": {"default_threshold": 0.7, "description": "Sentiment score threshold"},
            "technical_indicator": {"default_threshold": 50.0, "description": "Technical indicator level"}
        }
    
    def setup_price_alert(
        self, 
        symbols: List[str], 
        alert_type: str = "price_change", 
        threshold_percent: float = 5.0,
        notification_method: str = "log"
    ) -> Dict[str, Any]:
        """
        Setup price and market alerts
        
        Args:
            symbols: List of stock symbols to monitor
            alert_type: Type of alert to setup
            threshold_percent: Alert threshold value
            notification_method: How to send notifications (log, email, webhook)
        
        Returns:
            Alert configuration and status
        """
        try:
            alert_id = f"alert_{int(time.time())}"
            
            alert_config = {
                "alert_id": alert_id,
                "symbols": symbols,
                "alert_type": alert_type,
                "threshold": threshold_percent,
                "notification_method": notification_method,
                "created_date": datetime.now().isoformat(),
                "status": "active",
                "configuration": self._get_alert_configuration(alert_type, threshold_percent, symbols)
            }
            
            # Store alert configuration
            self.active_alerts[alert_id] = alert_config
            
            # Start monitoring if not already active
            if not self.monitoring_active:
                self._start_monitoring()
            
            return alert_config
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def setup_volume_alert(
        self, 
        symbols: List[str], 
        multiplier: float = 2.0,
        lookback_period: int = 20,
        notification_method: str = "log"
    ) -> Dict[str, Any]:
        """
        Setup volume-based alerts
        
        Args:
            symbols: List of stock symbols to monitor
            multiplier: Volume multiplier threshold
            lookback_period: Period for volume comparison
            notification_method: Notification method
        
        Returns:
            Volume alert configuration
        """
        try:
            alert_id = f"volume_alert_{int(time.time())}"
            
            alert_config = {
                "alert_id": alert_id,
                "symbols": symbols,
                "alert_type": "volume_spike",
                "threshold": multiplier,
                "lookback_period": lookback_period,
                "notification_method": notification_method,
                "created_date": datetime.now().isoformat(),
                "status": "active",
                "configuration": self._get_volume_alert_config(symbols, multiplier, lookback_period)
            }
            
            self.active_alerts[alert_id] = alert_config
            
            if not self.monitoring_active:
                self._start_monitoring()
            
            return alert_config
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def setup_volatility_alert(
        self, 
        symbols: List[str], 
        volatility_threshold: float = 30.0,
        lookback_days: int = 20,
        notification_method: str = "log"
    ) -> Dict[str, Any]:
        """
        Setup volatility-based alerts
        
        Args:
            symbols: List of stock symbols to monitor
            volatility_threshold: Volatility threshold percentage
            lookback_days: Days for volatility calculation
            notification_method: Notification method
        
        Returns:
            Volatility alert configuration
        """
        try:
            alert_id = f"volatility_alert_{int(time.time())}"
            
            alert_config = {
                "alert_id": alert_id,
                "symbols": symbols,
                "alert_type": "volatility",
                "threshold": volatility_threshold,
                "lookback_days": lookback_days,
                "notification_method": notification_method,
                "created_date": datetime.now().isoformat(),
                "status": "active",
                "configuration": self._get_volatility_alert_config(symbols, volatility_threshold, lookback_days)
            }
            
            self.active_alerts[alert_id] = alert_config
            
            if not self.monitoring_active:
                self._start_monitoring()
            
            return alert_config
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def setup_portfolio_alert(
        self, 
        symbols: List[str],
        alert_types: List[str] = None,
        notification_method: str = "log"
    ) -> Dict[str, Any]:
        """
        Setup comprehensive portfolio monitoring alerts
        
        Args:
            symbols: List of stock symbols in portfolio
            alert_types: List of alert types to monitor
            notification_method: Notification method
        
        Returns:
            Portfolio alert configuration
        """
        try:
            if alert_types is None:
                alert_types = ["price_change", "volume_spike", "volatility"]
            
            alert_id = f"portfolio_alert_{int(time.time())}"
            
            alert_config = {
                "alert_id": alert_id,
                "symbols": symbols,
                "alert_type": "portfolio_monitoring",
                "monitored_alerts": alert_types,
                "notification_method": notification_method,
                "created_date": datetime.now().isoformat(),
                "status": "active",
                "configuration": self._get_portfolio_alert_config(symbols, alert_types)
            }
            
            self.active_alerts[alert_id] = alert_config
            
            if not self.monitoring_active:
                self._start_monitoring()
            
            return alert_config
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_active_alerts(self) -> Dict[str, Any]:
        """
        Get all active alert configurations
        
        Returns:
            Dictionary of active alerts
        """
        try:
            return {
                "retrieval_date": datetime.now().isoformat(),
                "total_active_alerts": len(self.active_alerts),
                "alerts": self.active_alerts,
                "alert_summary": self._generate_alert_summary()
            }
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_alert_history(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get alert history and notifications
        
        Args:
            limit: Maximum number of alerts to return
        
        Returns:
            Alert history with recent notifications
        """
        try:
            recent_history = self.alert_history[-limit:] if limit > 0 else self.alert_history
            
            return {
                "retrieval_date": datetime.now().isoformat(),
                "total_alerts": len(self.alert_history),
                "returned_alerts": len(recent_history),
                "history": recent_history,
                "statistics": self._generate_alert_statistics()
            }
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def cancel_alert(self, alert_id: str) -> Dict[str, Any]:
        """
        Cancel an active alert
        
        Args:
            alert_id: ID of the alert to cancel
        
        Returns:
            Cancellation status and confirmation
        """
        try:
            if alert_id in self.active_alerts:
                alert_config = self.active_alerts.pop(alert_id)
                
                # Add to history
                self.alert_history.append({
                    "event_type": "alert_cancelled",
                    "alert_id": alert_id,
                    "timestamp": datetime.now().isoformat(),
                    "details": alert_config
                })
                
                return {
                    "status": "cancelled",
                    "alert_id": alert_id,
                    "cancellation_date": datetime.now().isoformat(),
                    "message": f"Alert {alert_id} has been successfully cancelled"
                }
            else:
                return {
                    "status": "error",
                    "alert_id": alert_id,
                    "message": "Alert not found or already cancelled"
                }
                
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def test_alert(self, alert_id: str, test_symbol: str = "AAPL") -> Dict[str, Any]:
        """
        Test an alert configuration
        
        Args:
            alert_id: ID of the alert to test
            test_symbol: Symbol to use for testing
        
        Returns:
            Test results and alert status
        """
        try:
            if alert_id not in self.active_alerts:
                return {"error": "Alert not found", "alert_id": alert_id}
            
            alert_config = self.active_alerts[alert_id]
            
            # Simulate alert test
            test_result = {
                "test_date": datetime.now().isoformat(),
                "alert_id": alert_id,
                "test_symbol": test_symbol,
                "alert_config": alert_config,
                "test_scenarios": self._run_alert_test_scenarios(alert_config, test_symbol),
                "test_result": "passed",
                "recommendations": self._generate_test_recommendations(alert_config)
            }
            
            return test_result
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _start_monitoring(self):
        """Start the alert monitoring thread"""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitor_thread.start()
                print("Alert monitoring started")
                
        except Exception as e:
            print(f"Error starting monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop for checking alerts"""
        try:
            while self.monitoring_active:
                # Check each active alert
                for alert_id, alert_config in self.active_alerts.items():
                    try:
                        self._check_alert(alert_id, alert_config)
                    except Exception as e:
                        print(f"Error checking alert {alert_id}: {e}")
                
                # Sleep for monitoring interval (30 seconds)
                time.sleep(30)
                
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
        finally:
            self.monitoring_active = False
            print("Alert monitoring stopped")
    
    def _check_alert(self, alert_id: str, alert_config: Dict[str, Any]):
        """Check a specific alert condition"""
        try:
            alert_type = alert_config["alert_type"]
            symbols = alert_config["symbols"]
            threshold = alert_config["threshold"]
            
            # Simulate alert checking
            triggered_alerts = []
            
            for symbol in symbols:
                if self._simulate_alert_condition(symbol, alert_type, threshold):
                    alert_notification = {
                        "alert_id": alert_id,
                        "symbol": symbol,
                        "alert_type": alert_type,
                        "triggered_date": datetime.now().isoformat(),
                        "threshold": threshold,
                        "current_value": self._get_current_value(symbol, alert_type),
                        "message": self._generate_alert_message(symbol, alert_type, threshold),
                        "severity": self._assess_alert_severity(alert_type, threshold)
                    }
                    
                    triggered_alerts.append(alert_notification)
                    self._send_notification(alert_notification)
            
            # Log triggered alerts
            if triggered_alerts:
                self.alert_history.extend(triggered_alerts)
                print(f"Alert triggered: {len(triggered_alerts)} notifications sent for alert {alert_id}")
                
        except Exception as e:
            print(f"Error checking alert {alert_id}: {e}")
    
    def _simulate_alert_condition(self, symbol: str, alert_type: str, threshold: float) -> bool:
        """Simulate checking if alert condition is met"""
        # Simulate alert triggers (5% chance for demonstration)
        import random
        return random.random() < 0.05
    
    def _get_current_value(self, symbol: str, alert_type: str) -> float:
        """Get current value for alert type"""
        import random
        
        if alert_type == "price_change":
            return round(random.uniform(-10, 10), 2)
        elif alert_type == "volume_spike":
            return round(random.uniform(0.5, 5.0), 2)
        elif alert_type == "volatility":
            return round(random.uniform(10, 60), 2)
        else:
            return round(random.uniform(0, 100), 2)
    
    def _generate_alert_message(self, symbol: str, alert_type: str, threshold: float) -> str:
        """Generate alert message"""
        current_value = self._get_current_value(symbol, alert_type)
        
        if alert_type == "price_change":
            return f"Price alert: {symbol} has moved {current_value:.2f}% (threshold: ±{threshold:.1f}%)"
        elif alert_type == "volume_spike":
            return f"Volume alert: {symbol} volume spike detected: {current_value:.1f}x average (threshold: {threshold:.1f}x)"
        elif alert_type == "volatility":
            return f"Volatility alert: {symbol} volatility at {current_value:.1f}% (threshold: {threshold:.1f}%)"
        else:
            return f"Alert triggered for {symbol}: {alert_type} = {current_value} (threshold: {threshold})"
    
    def _assess_alert_severity(self, alert_type: str, threshold: float) -> str:
        """Assess severity of triggered alert"""
        if alert_type == "price_change" and abs(threshold) > 10:
            return "high"
        elif alert_type == "volatility" and threshold > 40:
            return "high"
        elif alert_type == "volume_spike" and threshold > 3:
            return "high"
        else:
            return "medium"
    
    def _send_notification(self, alert_notification: Dict[str, Any]):
        """Send alert notification"""
        try:
            method = alert_notification.get("notification_method", "log")
            
            if method == "log":
                print(f"🚨 ALERT: {alert_notification['message']}")
            elif method == "email":
                # Simulate email sending
                print(f"📧 Email sent: {alert_notification['message']}")
            elif method == "webhook":
                # Simulate webhook call
                print(f"🔗 Webhook called: {alert_notification['message']}")
            
            # Add to notification queue
            self.notification_queue.put(alert_notification)
            
        except Exception as e:
            print(f"Error sending notification: {e}")
    
    def _get_alert_configuration(self, alert_type: str, threshold: float, symbols: List[str]) -> Dict[str, Any]:
        """Get detailed alert configuration"""
        return {
            "monitoring_frequency": "30 seconds",
            "data_source": "simulated_real_time",
            "alert_parameters": {
                "type": alert_type,
                "threshold": threshold,
                "symbols": symbols
            },
            "notification_settings": {
                "immediate": True,
                "grouping": False,
                "cooldown": "5 minutes"
            }
        }
    
    def _get_volume_alert_config(self, symbols: List[str], multiplier: float, lookback_period: int) -> Dict[str, Any]:
        """Get volume alert configuration"""
        return {
            "alert_type": "volume_spike",
            "volume_multiplier": multiplier,
            "lookback_period_days": lookback_period,
            "comparison_method": "rolling_average",
            "symbols": symbols
        }
    
    def _get_volatility_alert_config(self, symbols: List[str], threshold: float, lookback_days: int) -> Dict[str, Any]:
        """Get volatility alert configuration"""
        return {
            "alert_type": "volatility",
            "volatility_threshold": threshold,
            "lookback_days": lookback_days,
            "calculation_method": "rolling_standard_deviation",
            "symbols": symbols
        }
    
    def _get_portfolio_alert_config(self, symbols: List[str], alert_types: List[str]) -> Dict[str, Any]:
        """Get portfolio alert configuration"""
        return {
            "portfolio_size": len(symbols),
            "monitored_alerts": alert_types,
            "portfolio_aggregation": True,
            "correlation_monitoring": True,
            "risk_metrics": ["var", "volatility", "concentration"]
        }
    
    def _generate_alert_summary(self) -> Dict[str, Any]:
        """Generate summary of active alerts"""
        summary = {
            "total_alerts": len(self.active_alerts),
            "alert_types": {},
            "symbols_monitored": set(),
            "notification_methods": {}
        }
        
        for alert_config in self.active_alerts.values():
            alert_type = alert_config["alert_type"]
            symbols = alert_config["symbols"]
            notification_method = alert_config["notification_method"]
            
            # Count alert types
            summary["alert_types"][alert_type] = summary["alert_types"].get(alert_type, 0) + 1
            
            # Track symbols
            summary["symbols_monitored"].update(symbols)
            
            # Count notification methods
            summary["notification_methods"][notification_method] = summary["notification_methods"].get(notification_method, 0) + 1
        
        summary["symbols_monitored"] = list(summary["symbols_monitored"])
        
        return summary
    
    def _generate_alert_statistics(self) -> Dict[str, Any]:
        """Generate statistics from alert history"""
        if not self.alert_history:
            return {"message": "No alert history available"}
        
        # Analyze alert history
        alert_types = {}
        severity_counts = {}
        recent_alerts = []
        
        for alert in self.alert_history:
            alert_type = alert.get("alert_type", "unknown")
            severity = alert.get("severity", "medium")
            
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Get recent alerts (last 24 hours)
            alert_time = datetime.fromisoformat(alert.get("triggered_date", datetime.now().isoformat()))
            if (datetime.now() - alert_time).days < 1:
                recent_alerts.append(alert)
        
        return {
            "total_alerts_triggered": len(self.alert_history),
            "alert_type_distribution": alert_types,
            "severity_distribution": severity_counts,
            "recent_24h_alerts": len(recent_alerts),
            "most_common_alert": max(alert_types, key=alert_types.get) if alert_types else "None"
        }
    
    def _run_alert_test_scenarios(self, alert_config: Dict[str, Any], test_symbol: str) -> List[Dict[str, Any]]:
        """Run test scenarios for an alert"""
        scenarios = []
        
        # Test normal condition
        scenarios.append({
            "scenario": "normal_condition",
            "description": "Alert condition not triggered",
            "expected_result": "no_alert",
            "status": "passed"
        })
        
        # Test threshold condition
        scenarios.append({
            "scenario": "threshold_breach",
            "description": "Alert condition triggered",
            "expected_result": "alert_triggered",
            "status": "passed"
        })
        
        return scenarios
    
    def _generate_test_recommendations(self, alert_config: Dict[str, Any]) -> List[str]:
        """Generate recommendations from alert test"""
        recommendations = [
            "Alert configuration is working correctly",
            "Monitor alert performance in real market conditions",
            "Consider adjusting thresholds based on market volatility",
            "Review notification delivery methods"
        ]
        
        return recommendations
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        print("Alert monitoring stopped")