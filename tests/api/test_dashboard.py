def test_dashboard_stats_endpoint(client):
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "falls_today" in data
    assert "falls_week" in data
    assert "high_risk_persons" in data
    assert "analytics_enabled" in data
    assert "model" in data


def test_model_metrics_endpoint(client):
    response = client.get("/dashboard/model-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_metrics_enabled" in data
    assert "splits" in data
