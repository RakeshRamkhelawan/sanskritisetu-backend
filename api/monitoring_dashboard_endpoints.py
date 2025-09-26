"""
Monitoring Dashboard API Endpoints - TAAK 3.1 Implementation
FastAPI endpoints for real-time monitoring dashboard
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse

from core.monitoring.dashboard import get_monitoring_dashboard, MonitoringDashboard
from core.api.dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create router for dashboard endpoints
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Active WebSocket connections
active_connections: List[WebSocket] = []

class ConnectionManager:
    """Manage WebSocket connections for real-time dashboard updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Dashboard client disconnected. Active connections: {len(self.active_connections)}")

    async def send_to_all(self, data: Dict[str, Any]):
        """Send data to all connected clients"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send data to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_to_connection(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to specific connection"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Failed to send data to specific client: {e}")
            self.disconnect(websocket)

# Global connection manager
connection_manager = ConnectionManager()

@router.get("/health")
async def dashboard_health():
    """Health check endpoint for monitoring dashboard"""
    try:
        dashboard = await get_monitoring_dashboard()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_connections": len(connection_manager.active_connections),
            "dashboard_initialized": dashboard is not None
        }
    except Exception as e:
        logger.error(f"Dashboard health check failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard health check failed")

@router.get("/system-overview")
async def get_system_overview(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get complete system overview (REST endpoint)"""
    try:
        dashboard = await get_monitoring_dashboard()
        overview = await dashboard.get_system_overview()
        return overview
    except Exception as e:
        logger.error(f"Failed to get system overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system overview")

@router.get("/system-health")
async def get_system_health(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get system health status"""
    try:
        dashboard = await get_monitoring_dashboard()
        overview = await dashboard.get_system_overview()
        return overview.get('system_health', {})
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")

@router.get("/agents")
async def get_agent_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get all agent status information"""
    try:
        dashboard = await get_monitoring_dashboard()
        overview = await dashboard.get_system_overview()
        return overview.get('agent_status', [])
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve agent status")

@router.get("/performance")
async def get_performance_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get system performance metrics"""
    try:
        dashboard = await get_monitoring_dashboard()
        overview = await dashboard.get_system_overview()
        return {
            'performance_metrics': overview.get('performance_metrics', {}),
            'resource_usage': overview.get('resource_usage', {}),
            'database_health': overview.get('database_health', {})
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@router.get("/learning")
async def get_learning_progress(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get adaptive learning engine progress"""
    try:
        dashboard = await get_monitoring_dashboard()
        overview = await dashboard.get_system_overview()
        return overview.get('learning_progress', {})
    except Exception as e:
        logger.error(f"Failed to get learning progress: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve learning progress")

@router.get("/alerts")
async def get_recent_alerts(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get recent system alerts"""
    try:
        dashboard = await get_monitoring_dashboard()
        overview = await dashboard.get_system_overview()
        return overview.get('recent_alerts', [])
    except Exception as e:
        logger.error(f"Failed to get recent alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent alerts")

@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await connection_manager.connect(websocket)

    try:
        dashboard = await get_monitoring_dashboard()

        # Send initial data
        initial_data = await dashboard.get_system_overview()
        await connection_manager.send_to_connection(websocket, {
            "type": "initial_data",
            "data": initial_data
        })

        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for client message or timeout after 5 seconds
                await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                # Send periodic update every 5 seconds
                try:
                    updated_data = await dashboard.get_system_overview()
                    await connection_manager.send_to_connection(websocket, {
                        "type": "update",
                        "data": updated_data,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Failed to send periodic update: {e}")
                    break
            except Exception as e:
                logger.info(f"WebSocket receive error (client may have disconnected): {e}")
                break

    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)

@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time alert notifications"""
    await connection_manager.connect(websocket)

    try:
        # Send initial alert status
        await connection_manager.send_to_connection(websocket, {
            "type": "alert_status",
            "data": {"active_alerts": 0, "status": "monitoring"}
        })

        # Keep connection alive for alert notifications
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat every 30 seconds
                await connection_manager.send_to_connection(websocket, {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.info(f"Alert WebSocket receive error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("Alert WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Alert WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the monitoring dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sanskriti Setu - Monitoring Dashboard</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .status.healthy { background: #d4edda; color: #155724; }
            .status.warning { background: #fff3cd; color: #856404; }
            .status.critical { background: #f8d7da; color: #721c24; }
            .metric {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
            }
            .metric-value {
                font-weight: bold;
            }
            #connection-status {
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div id="connection-status" class="disconnected">Connecting...</div>

        <div class="dashboard">
            <div class="header">
                <h1>Sanskriti Setu - Monitoring Dashboard</h1>
                <p>Real-time system monitoring and health status</p>
                <div>Last Update: <span id="last-update">-</span></div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>System Health</h3>
                    <div id="system-health">
                        <div class="metric">
                            <span>Overall Status:</span>
                            <span id="overall-status" class="status">-</span>
                        </div>
                        <div class="metric">
                            <span>Database:</span>
                            <span id="db-status" class="status">-</span>
                        </div>
                        <div class="metric">
                            <span>Learning Engine:</span>
                            <span id="learning-status" class="status">-</span>
                        </div>
                        <div class="metric">
                            <span>Uptime:</span>
                            <span id="uptime" class="metric-value">-</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>Performance Metrics</h3>
                    <div id="performance-metrics">
                        <div class="metric">
                            <span>CPU Usage:</span>
                            <span id="cpu-usage" class="metric-value">-</span>
                        </div>
                        <div class="metric">
                            <span>Memory Usage:</span>
                            <span id="memory-usage" class="metric-value">-</span>
                        </div>
                        <div class="metric">
                            <span>Response Time:</span>
                            <span id="response-time" class="metric-value">-</span>
                        </div>
                        <div class="metric">
                            <span>Requests/min:</span>
                            <span id="requests-per-min" class="metric-value">-</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>Learning Progress</h3>
                    <div id="learning-progress">
                        <div class="metric">
                            <span>Events Processed:</span>
                            <span id="events-processed" class="metric-value">-</span>
                        </div>
                        <div class="metric">
                            <span>Insights Generated:</span>
                            <span id="insights-generated" class="metric-value">-</span>
                        </div>
                        <div class="metric">
                            <span>Learning Rate/hr:</span>
                            <span id="learning-rate" class="metric-value">-</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>Agent Status</h3>
                    <div id="agent-status">
                        <div>Loading agent information...</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let socket;
            let reconnectInterval;

            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/monitoring/ws/dashboard`;

                socket = new WebSocket(wsUrl);

                socket.onopen = function(event) {
                    console.log('WebSocket connected');
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').className = 'connected';
                    clearInterval(reconnectInterval);
                };

                socket.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    updateDashboard(message.data);
                };

                socket.onclose = function(event) {
                    console.log('WebSocket disconnected');
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').className = 'disconnected';

                    // Attempt to reconnect every 5 seconds
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                };

                socket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }

            function updateDashboard(data) {
                // Update timestamp
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

                // Update system health
                if (data.system_health) {
                    const health = data.system_health;
                    document.getElementById('overall-status').textContent = health.overall_status || '-';
                    document.getElementById('overall-status').className = `status ${health.overall_status || 'unknown'}`;

                    if (health.components) {
                        document.getElementById('db-status').textContent = health.components.database || '-';
                        document.getElementById('db-status').className = `status ${health.components.database || 'unknown'}`;

                        document.getElementById('learning-status').textContent = health.components.learning_engine || '-';
                        document.getElementById('learning-status').className = `status ${health.components.learning_engine || 'unknown'}`;
                    }

                    document.getElementById('uptime').textContent = health.uptime_hours ? `${health.uptime_hours.toFixed(1)}h` : '-';
                }

                // Update performance metrics
                if (data.performance_metrics) {
                    const perf = data.performance_metrics;
                    document.getElementById('cpu-usage').textContent = perf.cpu_usage_percent ? `${perf.cpu_usage_percent.toFixed(1)}%` : '-';
                    document.getElementById('memory-usage').textContent = perf.memory_usage_percent ? `${perf.memory_usage_percent.toFixed(1)}%` : '-';
                    document.getElementById('response-time').textContent = perf.average_response_time_ms ? `${perf.average_response_time_ms.toFixed(1)}ms` : '-';
                    document.getElementById('requests-per-min').textContent = perf.requests_per_minute ? perf.requests_per_minute.toFixed(1) : '-';
                }

                // Update learning progress
                if (data.learning_progress) {
                    const learning = data.learning_progress;
                    document.getElementById('events-processed').textContent = learning.events_processed || '0';
                    document.getElementById('insights-generated').textContent = learning.insights_generated || '0';
                    document.getElementById('learning-rate').textContent = learning.learning_rate_per_hour ? learning.learning_rate_per_hour.toFixed(1) : '0';
                }

                // Update agent status
                if (data.agent_status) {
                    const agentContainer = document.getElementById('agent-status');
                    if (data.agent_status.length > 0) {
                        agentContainer.innerHTML = data.agent_status.map(agent => `
                            <div class="metric">
                                <span>${agent.agent_id}:</span>
                                <span class="status ${agent.status}">${agent.status}</span>
                            </div>
                        `).join('');
                    } else {
                        agentContainer.innerHTML = '<div>No agents registered</div>';
                    }
                }
            }

            // Connect when page loads
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    return html_content

# Background task to broadcast updates to all connected clients
async def broadcast_updates():
    """Background task to send updates to all connected WebSocket clients"""
    while True:
        try:
            if connection_manager.active_connections:
                dashboard = await get_monitoring_dashboard()
                data = await dashboard.get_system_overview()
                await connection_manager.send_to_all({
                    "type": "broadcast_update",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })

            # Wait 10 seconds before next broadcast
            await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error in broadcast updates: {e}")
            await asyncio.sleep(10)

# Export the router
__all__ = ['router', 'connection_manager', 'broadcast_updates']