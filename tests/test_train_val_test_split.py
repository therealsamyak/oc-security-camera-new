#!/usr/bin/env python3
"""Test train/validation/test splitting functionality."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_custom_controller import CustomController
from generate_training_data import load_power_profiles, generate_training_scenarios


def test_data_splitting():
    """Test train/validation/test data splitting."""
    print("🔀 Testing Data Splitting...")

    # Create mock training data
    mock_data = []
    for i in range(100):
        mock_data.append(
            {
                "battery_level": np.random.randint(5, 100),
                "clean_energy_percentage": np.random.randint(0, 100),
                "accuracy_requirement": np.random.uniform(0.3, 1.0),
                "latency_requirement": np.random.choice([1, 2, 3, 5, 8, 10, 15, 20]),
                "optimal_model": "YOLOv10_N",
                "should_charge": np.random.choice([True, False]),
            }
        )

    controller = CustomController()
    train_data, val_data, test_data = controller.split_data(mock_data)

    # Check splits
    total_len = len(mock_data)
    assert len(train_data) + len(val_data) + len(test_data) == total_len
    assert abs(len(train_data) / total_len - 0.7) < 0.05
    assert abs(len(val_data) / total_len - 0.2) < 0.05
    assert abs(len(test_data) / total_len - 0.1) < 0.05

    print(
        f"✅ Data split correct: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
    )
    return train_data, val_data, test_data


def test_evaluation():
    """Test model evaluation."""
    print("📊 Testing Evaluation...")

    controller = CustomController()

    # Create mock data
    mock_data = [
        {
            "battery_level": 50,
            "clean_energy_percentage": 75,
            "accuracy_requirement": 0.8,
            "latency_requirement": 10,
            "optimal_model": "YOLOv10_X",
            "should_charge": True,
        },
        {
            "battery_level": 25,
            "clean_energy_percentage": 25,
            "accuracy_requirement": 0.4,
            "latency_requirement": 5,
            "optimal_model": "YOLOv10_N",
            "should_charge": False,
        },
    ]

    mock_models = {
        "YOLOv10_N": {"accuracy": 39.5, "latency": 1.56, "power_cost": 602.25},
        "YOLOv10_X": {"accuracy": 54.4, "latency": 12.2, "power_cost": 3476.75},
    }

    metrics = controller.evaluate(mock_data, mock_models)

    assert "loss" in metrics
    assert "model_accuracy" in metrics
    assert "charge_accuracy" in metrics
    assert "overall_accuracy" in metrics

    print(
        f"✅ Evaluation working: Loss={metrics['loss']:.4f}, Overall Acc={metrics['overall_accuracy']:.3f}"
    )
    return metrics


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
    assert len(scenarios) <= 150000, "Too many scenarios generated"

    # Check first scenario
    battery, clean_energy, acc_req, lat_req = scenarios[0]
    assert 5 <= battery <= 100, f"Invalid battery level: {battery}"
    assert 0 <= clean_energy <= 100, f"Invalid clean energy: {clean_energy}"
    assert 0.2 <= acc_req <= 1.0, f"Invalid accuracy requirement: {acc_req}"
    assert 1 <= lat_req <= 30, f"Invalid latency requirement: {lat_req}"

    print(f"✓ Generated {len(scenarios)} scenarios successfully")
    return scenarios


def main():
    """Run splitting tests."""
    print("🧪 Testing Train/Val/Test Splitting...")

    test_load_power_profiles()
    test_generate_training_scenarios()
    train_data, val_data, test_data = test_data_splitting()
    test_evaluation()

    print("\n✅ All splitting tests passed!")
    print("📋 Training now includes proper train/validation/test partitioning")


if __name__ == "__main__":
    main()
