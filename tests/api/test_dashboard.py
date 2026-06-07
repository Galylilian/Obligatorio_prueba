def test_dashboard_stats_endpoint(client):
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "falls_today" in data
    assert "falls_week" in data
    assert "high_risk_persons" in data
    assert "analytics_enabled" in data
