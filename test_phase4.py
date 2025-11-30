#!/usr/bin/env python3
"""
Simple test script to verify Phase 4 implementation works.
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_basic_functionality():
    """Test basic functionality of Phase 4 components."""
    print("Testing Phase 4 implementation...")

    try:
        # Test 1: Load power profiles
        print("1. Loading power profiles...")
        with open("results/power_profiles.json", "r") as f:
            power_profiles = json.load(f)
        print(f"   Loaded {len(power_profiles)} model profiles")

        # Test 2: Test SimulationConfig
        print("2. Testing SimulationConfig...")
        from simulation_engine import SimulationConfig

        config = SimulationConfig(duration_days=1, task_interval_seconds=5)
        print(
            f"   Config: {config.duration_days} days, {config.task_interval_seconds}s interval"
        )

        # Test 3: Test Battery with Wh units
        print("3. Testing Battery (Wh units)...")
        from battery import Battery

        battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)
        print(
            f"   Battery: {battery.get_level_wh():.2f}Wh / {battery.capacity_wh:.2f}Wh"
        )
        print(f"   Percentage: {battery.get_percentage():.1f}%")

        # Test 4: Test Controllers
        print("4. Testing Controllers...")
        from controller import NaiveWeakController, CustomController

        weak_controller = NaiveWeakController()
        CustomController()

        # Test model selection
        available_models = {}
        for name, profile in power_profiles.items():
            available_models[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"] * 1000,
                "power_cost": profile["model_power_mw"],
            }

        weak_choice = weak_controller.select_model(
            battery_level=50.0,
            clean_energy_percentage=30.0,
            user_accuracy_requirement=80.0,
            user_latency_requirement=2000.0,
            available_models=available_models,
        )
        print(f"   Weak controller chose: {weak_choice.model_name}")

        # Test 5: Test TaskGenerator
        print("5. Testing TaskGenerator...")
        from simulation_engine import TaskGenerator

        task_gen = TaskGenerator(seed=42)
        task = task_gen.generate_task(0.0)
        if task:
            print(
                f"   Generated task: accuracy={task.accuracy_requirement:.1f}%, latency={task.latency_requirement:.0f}ms"
            )
        else:
            print("   No task generated (10% chance)")

        # Test 6: Test ConfigLoader
        print("6. Testing ConfigLoader...")
        from config_loader import ConfigLoader

        config_loader = ConfigLoader("config.jsonc")
        sim_config = config_loader.get_simulation_config()
        print(
            f"   Loaded config: {sim_config.duration_days} days, locations: {sim_config.locations}"
        )

        # Test 7: Test MetricsCollector
        print("7. Testing MetricsCollector...")
        from metrics_collector import MetricsCollector, CSVExporter

        collector = MetricsCollector()
        collector.start_simulation()

        # Simulate some task data
        task_data = {
            "completed": True,
            "energy_used_wh": 0.001,
            "clean_energy_used_wh": 0.0005,
            "model_used": "YOLOv10_N",
        }
        collector.update_task_metrics(task_data)
        collector.update_battery_level(0.0, 95.0)
        collector.calculate_final_metrics()

        metrics = collector.get_metrics()
        print(
            f"   Metrics: {metrics['total_tasks']} tasks, {metrics['task_completion_rate']:.1f}% completion"
        )

        # Test 8: Test CSVExporter
        print("8. Testing CSVExporter...")
        exporter = CSVExporter("results")
        simulation_info = {
            "id": "test",
            "location": "CA",
            "season": "summer",
            "controller": "naive_weak",
        }

        # Add final battery level to metrics
        metrics["final_battery_level"] = 95.0
        metrics["avg_battery_level"] = 95.0

        summary_file = exporter.export_summary(metrics, simulation_info)
        if summary_file:
            print(f"   Summary exported to: {summary_file}")

        print("\n✅ All Phase 4 components working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_short_simulation():
    """Test a very short simulation run."""
    print("\nTesting short simulation run (1 minute)...")

    try:
        from simulation_engine import SimulationEngine, SimulationConfig
        from controller import NaiveWeakController

        # Create minimal config for quick test
        config = SimulationConfig(
            duration_days=0,  # Will be overridden
            task_interval_seconds=5,
            time_acceleration=100,  # Fast acceleration
            battery_capacity_wh=5.0,
            charge_rate_watts=100.0,
        )

        # Override duration for very short test
        config.duration_days = 0.0007  # ~1 minute

        # Load power profiles
        with open("results/power_profiles.json", "r") as f:
            power_profiles = json.load(f)

        # Create simulation
        controller = NaiveWeakController()
        engine = SimulationEngine(
            config=config,
            controller=controller,
            location="CA",
            season="summer",
            week=1,
            power_profiles=power_profiles,
        )

        # Run simulation
        metrics = engine.run()

        print("   Simulation completed:")
        print(f"   Total tasks: {metrics['total_tasks']}")
        print(f"   Completed tasks: {metrics['completed_tasks']}")
        print(f"   Task completion rate: {metrics['task_completion_rate']:.1f}%")
        energy_used = metrics.get("total_energy_wh", 0)
        clean_energy_pct = metrics.get("clean_energy_percentage", 0)
        print(f"   Energy used: {energy_used:.4f} Wh")
        print(f"   Clean energy: {clean_energy_pct:.1f}%")

        print("✅ Short simulation test passed!")
        return True

    except Exception as e:
        print(f"❌ Short simulation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4 IMPLEMENTATION TEST")
    print("=" * 60)

    success1 = test_basic_functionality()
    success2 = test_short_simulation()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! Phase 4 implementation is working.")
        print("\nNext steps:")
        print("1. Run full simulation: python simulation_runner.py")
        print("2. Check results in results/ directory")
        print("3. Review performance metrics")
    else:
        print("❌ SOME TESTS FAILED. Check the errors above.")
    print("=" * 60)

    sys.exit(0 if (success1 and success2) else 1)
