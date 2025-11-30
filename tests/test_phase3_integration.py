#!/usr/bin/env python3
"""Integration test for complete Phase 3 workflow."""

import os


def test_phase3_workflow():
    """Test that Phase 3 workflow runs end-to-end."""
    print("🧪 Testing Phase 3 Integration Workflow...")

    # Check required files exist
    required_files = [
        "generate_training_data.py",
        "train_custom_controller.py",
        "results/power_profiles.json",
        "model-data/model-data.csv",
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return False
    print("✅ All required files exist")

    # Test training data generation (small sample)
    print("\n📊 Testing training data generation...")
    try:
        # Create a small test version
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                """
import sys
sys.path.insert(0, '.')
from generate_training_data import load_power_profiles, generate_training_scenarios

# Test loading
models = load_power_profiles()
print(f"Loaded {len(models)} models")

# Test scenario generation (small sample)
scenarios = generate_training_scenarios()
print(f"Generated {len(scenarios)} scenarios")

# Test first scenario
from generate_training_data import solve_mips_scenario
battery, clean_energy, acc_req, lat_req = scenarios[0]
selected_model, should_charge = solve_mips_scenario(
    battery, clean_energy, acc_req, lat_req, models
)
print(f"Sample scenario: model={selected_model}, charge={should_charge}")
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"❌ Training data generation failed: {result.stderr}")
            return False

        print("✅ Training data generation working")
        print(result.stdout)

    except Exception as e:
        print(f"❌ Training data generation error: {e}")
        return False

    # Test custom controller training (small sample)
    print("\n🤖 Testing custom controller training...")
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                """
import sys
sys.path.insert(0, '.')
from train_custom_controller import CustomController, load_power_profiles

# Test loading
models = load_power_profiles()
print(f"Loaded {len(models)} models")

# Test controller
controller = CustomController()
print("Controller initialized")

# Test feature extraction
scenario = {
    "battery_level": 50,
    "clean_energy_percentage": 75,
    "accuracy_requirement": 0.8,
    "latency_requirement": 1500,
    "optimal_model": "YOLOv10_X",
    "should_charge": True
}
features = controller.extract_features(scenario)
print(f"Features extracted: {features}")

# Test prediction
import numpy as np
model, charge = controller.predict_model_and_charge(
    features, list(models.keys()), models
)
print(f"Prediction: model={model}, charge={charge}")

# Test training step
loss = controller.train_step(scenario, models, learning_rate=0.01)
print(f"Training step loss: {loss:.4f}")
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"❌ Custom controller training failed: {result.stderr}")
            return False

        print("✅ Custom controller training working")
        print(result.stdout)

    except Exception as e:
        print(f"❌ Custom controller training error: {e}")
        return False

    # Check output directories
    print("\n📁 Checking output structure...")
    os.makedirs("results", exist_ok=True)

    expected_outputs = [
        "results/training_data.json",
        "results/custom_controller_weights.json",
    ]

    print("✅ Output directories ready")
    print(f"📂 Will create: {expected_outputs}")

    return True


def main():
    """Run integration test."""
    print("=" * 60)
    print("PHASE 3 INTEGRATION TEST")
    print("=" * 60)

    if test_phase3_workflow():
        print("\n🎉 Phase 3 integration test PASSED!")
        print("\n📋 Ready to run:")
        print("   1. uv run python generate_training_data.py")
        print("   2. uv run python train_custom_controller.py")
        print("\n✨ This will complete Phase 3 of the plan!")
    else:
        print("\n❌ Phase 3 integration test FAILED!")
        print("🔧 Fix issues before running full workflow")


if __name__ == "__main__":
    main()
