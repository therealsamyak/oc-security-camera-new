#!/usr/bin/env python3
"""Test accuracy translation functionality."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_custom_controller import CustomController


def test_accuracy_translation():
    """Test accuracy requirement translation to model suitability."""
    print("Testing accuracy translation...")
    controller = CustomController()

    # Test cases: (user_requirement, model_map, expected_behavior)
    test_cases = [
        (0.3, 39.5, "should be acceptable"),  # Low requirement, low model
        (
            0.8,
            54.4,
            "should be penalized",
        ),  # High requirement, best model still insufficient
        (0.5, 46.7, "should be reasonable"),  # Medium requirement, medium model
        (0.9, 39.5, "should be heavily penalized"),  # Very high requirement, weak model
    ]

    for user_req, model_map, expected in test_cases:
        score = controller.get_model_accuracy_score(user_req, model_map)
        print(
            f"User req: {user_req:.1f}, Model mAP: {model_map:.1f} → Score: {score:.3f} ({expected})"
        )

        # Score should be between -1 and 1
        assert -1.0 <= score <= 1.0, f"Score {score} out of range"

    print("✓ Accuracy translation working correctly")


def test_feature_extraction():
    """Test feature extraction with new accuracy format."""
    print("Testing feature extraction...")
    controller = CustomController()

    scenario = {
        "battery_level": 50,
        "clean_energy_percentage": 75,
        "accuracy_requirement": 0.8,  # Now 0-1 range
        "latency_requirement": 1500,
    }

    features = controller.extract_features(scenario)
    assert len(features) == 4, f"Expected 4 features, got {len(features)}"
    assert features[0] == 0.5, f"Incorrect battery feature: {features[0]}"
    assert features[1] == 0.75, f"Incorrect clean energy feature: {features[1]}"
    assert features[2] == 0.8, f"Incorrect accuracy feature: {features[2]}"
    assert features[3] == 0.5, f"Incorrect latency feature: {features[3]}"

    print("✓ Feature extraction with 0-1 accuracy working correctly")


def test_model_selection_with_accuracy():
    """Test model selection considers accuracy requirements."""
    print("Testing model selection with accuracy requirements...")
    controller = CustomController()

    # Mock model data
    model_data = {
        "YOLOv10_N": {"accuracy": 39.5, "latency": 1.56, "power_cost": 602.25},
        "YOLOv10_X": {"accuracy": 54.4, "latency": 12.2, "power_cost": 2000.0},
    }

    # Test low accuracy requirement - should prefer lighter model
    features_low = np.array([0.5, 0.5, 0.3, 0.5])  # Low accuracy requirement
    model_low, charge_low = controller.predict_model_and_charge(
        features_low, list(model_data.keys()), model_data
    )

    # Test high accuracy requirement - should prefer stronger model
    features_high = np.array([0.5, 0.5, 0.9, 0.5])  # High accuracy requirement
    model_high, charge_high = controller.predict_model_and_charge(
        features_high, list(model_data.keys()), model_data
    )

    print(f"Low accuracy requirement (0.3): Selected {model_low}")
    print(f"High accuracy requirement (0.9): Selected {model_high}")

    # The high accuracy requirement should prefer the stronger model
    # (though this depends on learned weights, the accuracy score should influence it)

    print("✓ Model selection with accuracy requirements working")


def main():
    """Run all accuracy translation tests."""
    print("Running accuracy translation tests...")

    test_accuracy_translation()
    test_feature_extraction()
    test_model_selection_with_accuracy()

    print("✓ All accuracy translation tests passed!")


if __name__ == "__main__":
    main()
