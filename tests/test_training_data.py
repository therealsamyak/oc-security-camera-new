#!/usr/bin/env python3
"""Test training data generation functionality."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_training_data import load_power_profiles, generate_training_scenarios


def test_load_power_profiles():
    """Test loading power profiles."""
    print("Testing load_power_profiles...")
    models = load_power_profiles()

    assert len(models) > 0, "No models loaded"
    assert "YOLOv10_N" in models, "YOLOv10_N not found"

    for model_name, data in models.items():
        assert "accuracy" in data, f"Missing accuracy for {model_name}"
        assert "latency" in data, f"Missing latency for {model_name}"
        assert "power_cost" in data, f"Missing power_cost for {model_name}"

    print(f"✓ Loaded {len(models)} models successfully")
    return models


def test_generate_training_scenarios():
    """Test scenario generation."""
    print("Testing generate_training_scenarios...")
    scenarios = generate_training_scenarios()

    assert len(scenarios) > 0, "No scenarios generated"
    assert len(scenarios) <= 10000, "Too many scenarios generated"

    # Check first scenario
    battery, clean_energy, acc_req, lat_req = scenarios[0]
    assert 5 <= battery <= 100, f"Invalid battery level: {battery}"
    assert 0 <= clean_energy <= 100, f"Invalid clean energy: {clean_energy}"
    assert 70 <= acc_req <= 95, f"Invalid accuracy requirement: {acc_req}"
    assert 1000 <= lat_req <= 3000, f"Invalid latency requirement: {lat_req}"

    print(f"✓ Generated {len(scenarios)} scenarios successfully")
    return scenarios


def main():
    """Run all tests."""
    print("Running training data generation tests...")

    test_load_power_profiles()
    test_generate_training_scenarios()

    print("✓ All tests passed!")


if __name__ == "__main__":
    main()
