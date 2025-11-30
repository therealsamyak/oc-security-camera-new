#!/usr/bin/env python3
"""Test custom controller training functionality."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_custom_controller import CustomController, load_power_profiles


def test_custom_controller_init():
    """Test CustomController initialization."""
    print("Testing CustomController initialization...")
    controller = CustomController()

    assert "accuracy_weight" in controller.weights, "Missing accuracy_weight"
    assert "latency_weight" in controller.weights, "Missing latency_weight"
    assert "clean_energy_weight" in controller.weights, "Missing clean_energy_weight"
    assert controller.charge_threshold == 0.5, "Incorrect default charge threshold"

    print("✓ CustomController initialized successfully")
    return controller


def test_feature_extraction():
    """Test feature extraction from scenarios."""
    print("Testing feature extraction...")
    controller = CustomController()

    scenario = {
        "battery_level": 50,
        "clean_energy_percentage": 75,
        "accuracy_requirement": 0.85,  # Now 0-1 range
        "latency_requirement": 1500,
    }

    features = controller.extract_features(scenario)
    assert len(features) == 4, f"Expected 4 features, got {len(features)}"
    assert features[0] == 0.5, f"Incorrect battery feature: {features[0]}"
    assert features[1] == 0.75, f"Incorrect clean energy feature: {features[1]}"
    assert features[2] == 0.85, f"Incorrect accuracy feature: {features[2]}"
    assert features[3] == 0.5, f"Incorrect latency feature: {features[3]}"

    print("✓ Feature extraction successful")


def test_prediction():
    """Test model and charging prediction."""
    print("Testing prediction...")
    controller = CustomController()

    features = np.array([0.5, 0.75, 0.85, 0.5])
    available_models = ["YOLOv10_N", "YOLOv10_S"]

    # Mock model data for prediction
    model_data = {
        "YOLOv10_N": {"accuracy": 39.5, "latency": 1.56, "power_cost": 602.25},
        "YOLOv10_S": {"accuracy": 46.7, "latency": 2.66, "power_cost": 800.0},
    }

    model, charge = controller.predict_model_and_charge(
        features, available_models, model_data
    )

    assert model in available_models, f"Predicted model {model} not in available models"
    assert isinstance(charge, (bool, np.bool_)), (
        f"Charge decision should be boolean, got {type(charge)}"
    )

    print("✓ Prediction successful")


def test_load_power_profiles():
    """Test loading power profiles for training."""
    print("Testing load_power_profiles...")
    models = load_power_profiles()

    assert len(models) > 0, "No models loaded"

    for model_name, data in models.items():
        assert "accuracy" in data, f"Missing accuracy for {model_name}"
        assert "latency" in data, f"Missing latency for {model_name}"
        assert "power_cost" in data, f"Missing power_cost for {model_name}"

    print(f"✓ Loaded {len(models)} models successfully")
    return models


def main():
    """Run all tests."""
    print("Running custom controller training tests...")

    test_custom_controller_init()
    test_feature_extraction()
    test_prediction()
    test_load_power_profiles()

    print("✓ All tests passed!")


if __name__ == "__main__":
    main()
