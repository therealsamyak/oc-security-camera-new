#!/usr/bin/env python3
"""
Test improved CustomController training algorithm.
"""

import json
import sys
import os

# Add parent directory to path to import train_custom_controller
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train_custom_controller import CustomController


def load_power_profiles() -> dict:
    """Load power profiles from results and real model data."""
    with open("results/power_profiles.json", "r") as f:
        profiles = json.load(f)

    # Load real model data
    model_data = {}
    with open("model-data/model-data.csv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(",")
            model = parts[0].strip('"')
            version = parts[1].strip('"')
            latency = float(parts[2].strip('"'))
            accuracy = float(parts[3].strip('"'))
            model_data[f"{model}_{version}"] = {
                "accuracy": accuracy,
                "latency": latency,
            }

    models = {}
    for model_name, data in profiles.items():
        real_data = model_data.get(model_name, {})
        models[model_name] = {
            "accuracy": real_data.get("accuracy", 85.0),
            "latency": real_data.get(
                "latency", data["avg_inference_time_seconds"] * 1000
            ),
            "power_cost": data["model_power_mw"],
        }

    return models


def create_small_training_data() -> list:
    """Create small training dataset for testing."""
    models = load_power_profiles()
    training_data = []

    # Generate diverse scenarios
    for battery in [20, 50, 80]:
        for clean_energy in [10, 50, 90]:
            for acc_req in [0.4, 0.7, 0.9]:
                for lat_req in [5, 10, 15]:
                    # Simple heuristic for optimal model
                    suitable_models = [
                        m
                        for m in models.keys()
                        if models[m]["accuracy"] >= acc_req * 100
                        and models[m]["latency"] <= lat_req
                    ]

                    if suitable_models:
                        # Choose most accurate suitable model
                        optimal_model = max(
                            suitable_models, key=lambda x: models[x]["accuracy"]
                        )
                        should_charge = battery < 30 or (
                            clean_energy > 80 and battery < 70
                        )

                        training_data.append(
                            {
                                "battery_level": battery,
                                "clean_energy_percentage": clean_energy,
                                "accuracy_requirement": acc_req,
                                "latency_requirement": lat_req,
                                "optimal_model": optimal_model,
                                "should_charge": should_charge,
                            }
                        )

    return training_data


def test_improved_training():
    """Test the improved training algorithm."""
    print("Creating small training dataset...")
    training_data = create_small_training_data()
    print(f"Created {len(training_data)} training samples")

    print("Loading power profiles...")
    available_models = load_power_profiles()

    print("Initializing CustomController...")
    controller = CustomController()

    print("Testing improved training...")
    test_metrics = controller.train(
        training_data, available_models, epochs=50, learning_rate=0.05
    )

    print("\nResults:")
    print(f"Model Accuracy: {test_metrics['model_accuracy']:.3f}")
    print(f"Charge Accuracy: {test_metrics['charge_accuracy']:.3f}")
    print(f"Overall Accuracy: {test_metrics['overall_accuracy']:.3f}")

    # Test if improvement is significant
    if test_metrics["overall_accuracy"] > 0.6:
        print("✅ Training algorithm improved significantly!")
        return True
    else:
        print("❌ Training still needs improvement")
        return False


if __name__ == "__main__":
    test_improved_training()
