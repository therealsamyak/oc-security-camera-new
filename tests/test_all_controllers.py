#!/usr/bin/env python3
"""
Integration test for all 4 controller types.
Runs 1-day simulation for each controller to verify functionality.
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation_engine import SimulationEngine, SimulationConfig
from src.controller import (
    NaiveWeakController,
    NaiveStrongController,
    OracleController,
    CustomController,
)

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), "..")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def load_power_profiles():
    """Load power profiles for testing."""
    with open("results/power_profiles.json", "r") as f:
        return json.load(f)


def test_controller(controller_class, controller_name, power_profiles, test_data=None):
    """Test a single controller."""
    print(f"\n🧪 Testing {controller_name}...")
    print(f"   Controller class: {controller_class}")

    # Initialize controller
    try:
        if controller_class == OracleController:
            # Oracle needs future data - use dummy data for testing
            print("   Initializing OracleController with dummy future data...")
            controller = controller_class(future_energy_data={}, future_tasks=100)
        elif controller_class == CustomController:
            weights_file = "results/custom_controller_weights.json"
            print(
                f"   Initializing CustomController with weights from {weights_file}..."
            )
            controller = controller_class(weights_file)
            print("   CustomController loaded successfully")
            print(
                f"   Model weights keys: {list(controller.model_weights.keys()) if hasattr(controller, 'model_weights') else 'None'}"
            )
            print(
                f"   Charge weights: {controller.charge_weights if hasattr(controller, 'charge_weights') else 'None'}"
            )
            print(
                f"   Charge threshold: {controller.charge_threshold if hasattr(controller, 'charge_threshold') else 'None'}"
            )
        else:
            print(f"   Initializing {controller_name}...")
            controller = controller_class()
    except Exception as e:
        print(f"❌ {controller_name} initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Configure simulation for 1 day
    print("   Configuring simulation...")
    config = SimulationConfig(
        duration_days=1,
        task_interval_seconds=60,  # 1 minute intervals for faster testing
        time_acceleration=1000,  # Speed up testing
        battery_capacity_wh=5.0,
        charge_rate_watts=100.0,
        locations=["CA"],  # Single location for testing
        seasons=["summer"],  # Single season for testing
    )
    print(
        f"   Simulation config: {config.duration_days} days, {config.task_interval_seconds}s intervals"
    )

    # Run simulation
    try:
        print("   Creating simulation engine...")
        engine = SimulationEngine(
            config=config,
            controller=controller,
            location="CA",
            season="summer",
            week=1,
            power_profiles=power_profiles,
        )
        print("   Simulation engine created successfully")
        print(f"   Battery capacity: {engine.battery.capacity_wh} Wh")
        print(f"   Battery current level: {engine.battery.get_percentage():.1f}%")
        print(f"   Available models: {list(engine.power_profiles.keys())}")

        print("   Starting simulation run...")
        metrics = engine.run()
        print("   Simulation completed")

        # Detailed metrics logging
        print(f"   📊 Raw metrics keys: {list(metrics.keys())}")
        print(f"   📊 Total tasks: {metrics.get('total_tasks', 'N/A')}")
        print(f"   📊 Completed tasks: {metrics.get('completed_tasks', 'N/A')}")
        print(f"   📊 Missed deadlines: {metrics.get('missed_deadlines', 'N/A')}")
        print(f"   📊 Total energy (Wh): {metrics.get('total_energy_wh', 'N/A')}")
        print(f"   📊 Clean energy (Wh): {metrics.get('clean_energy_wh', 'N/A')}")
        print(f"   📊 Clean energy %: {metrics.get('clean_energy_percentage', 'N/A')}")
        print(f"   📊 Model selections: {metrics.get('model_selections', 'N/A')}")
        print(f"   📊 Battery levels count: {len(metrics.get('battery_levels', []))}")

        # Basic validation
        assert metrics["total_tasks"] > 0, f"{controller_name}: No tasks generated"
        assert metrics["completed_tasks"] >= 0, (
            f"{controller_name}: Negative completed tasks"
        )
        assert metrics["total_energy_wh"] >= 0, (
            f"{controller_name}: Negative energy usage"
        )
        assert 0 <= metrics["clean_energy_percentage"] <= 100, (
            f"{controller_name}: Invalid clean energy percentage"
        )

        print(f"✅ {controller_name}:")
        print(
            f"   Tasks: {metrics['total_tasks']} total, {metrics['completed_tasks']} completed"
        )
        print(f"   Energy: {metrics['total_energy_wh']:.4f} Wh")
        print(f"   Clean Energy: {metrics['clean_energy_percentage']:.1f}%")
        print(f"   Task Completion: {metrics.get('task_completion_rate', 0):.1f}%")

        # Model selection breakdown
        model_selections = metrics.get("model_selections", {})
        if model_selections:
            print("   Model usage:")
            for model, count in model_selections.items():
                if count > 0:
                    print(f"     {model}: {count} times")

        return True

    except Exception as e:
        print(f"❌ {controller_name} simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test all 4 controller types."""
    print("🚀 Starting controller integration tests...")

    # Load required data
    try:
        power_profiles = load_power_profiles()
        print(f"✅ Loaded {len(power_profiles)} model profiles")
    except FileNotFoundError:
        print("❌ power_profiles.json not found. Run benchmark_power.py first.")
        return False

    # Test all controllers
    controllers = [
        (NaiveWeakController, "NaiveWeakController"),
        (NaiveStrongController, "NaiveStrongController"),
        (OracleController, "OracleController"),
        (CustomController, "CustomController"),
    ]

    results = []
    for controller_class, controller_name in controllers:
        success = test_controller(controller_class, controller_name, power_profiles)
        results.append((controller_name, success))

    # Summary
    print("\n📊 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for controller_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {controller_name}")

    print(f"\n🎯 Overall: {passed}/{total} controllers working")

    if passed == total:
        print("🎉 All controllers are working correctly!")
        return True
    else:
        print("⚠️  Some controllers have issues. Check logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
