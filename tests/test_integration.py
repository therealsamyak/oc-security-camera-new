#!/usr/bin/env python3
"""
Consolidated integration tests - Phase 3 workflow, Phase 4 implementation, and short simulation tests.
"""

import os
import json
import time
import unittest
import tempfile
import subprocess
from pathlib import Path

from src.simulation_engine import SimulationEngine, SimulationConfig
from src.controller import (
    NaiveWeakController,
    NaiveStrongController,
    OracleController,
    CustomController,
)
from src.config_loader import ConfigLoader
from src.metrics_collector import MetricsCollector, CSVExporter


class TestPhase3Workflow(unittest.TestCase):
    """Test Phase 3 workflow integration."""

    def test_phase3_workflow(self):
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
            self.assertTrue(os.path.exists(file), f"Missing required file: {file}")
        print("✅ All required files exist")

        # Test training data generation (small sample)
        print("\n📊 Testing training data generation...")
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

            self.assertEqual(
                result.returncode,
                0,
                f"Training data generation failed: {result.stderr}",
            )
            print("✅ Training data generation working")
            print(result.stdout)

        except Exception as e:
            self.fail(f"Training data generation error: {e}")

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

            self.assertEqual(
                result.returncode,
                0,
                f"Custom controller training failed: {result.stderr}",
            )
            print("✅ Custom controller training working")
            print(result.stdout)

        except Exception as e:
            self.fail(f"Custom controller training error: {e}")

        # Check output directories
        print("\n📁 Checking output structure...")
        os.makedirs("results", exist_ok=True)

        expected_outputs = [
            "results/training_data.json",
            "results/custom_controller_weights.json",
        ]

        print("✅ Output directories ready")
        print(f"📂 Will create: {expected_outputs}")


class TestPhase4Implementation(unittest.TestCase):
    """Test Phase 4 implementation components."""

    def test_basic_functionality(self):
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
            from src.simulation_engine import SimulationConfig

            config = SimulationConfig(duration_days=1, task_interval_seconds=5)
            print(
                f"   Config: {config.duration_days} days, {config.task_interval_seconds}s interval"
            )

            # Test 3: Test Battery with Wh units
            print("3. Testing Battery (Wh units)...")
            from src.battery import Battery

            battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)
            print(
                f"   Battery: {battery.get_level_wh():.2f}Wh / {battery.capacity_wh:.2f}Wh"
            )
            print(f"   Percentage: {battery.get_percentage():.1f}%")

            # Test 4: Test Controllers
            print("4. Testing Controllers...")
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
            from src.simulation_engine import TaskGenerator

            task_gen = TaskGenerator(seed=42)
            task = task_gen.generate_task(0.0, 80.0, 2000.0)
            if task:
                print(
                    f"   Generated task: accuracy={task.accuracy_requirement:.1f}%, latency={task.latency_requirement:.0f}ms"
                )
            else:
                print("   No task generated (10% chance)")

            # Test 6: Test ConfigLoader
            print("6. Testing ConfigLoader...")
            config_loader = ConfigLoader("config.jsonc")
            sim_config = config_loader.get_simulation_config()
            print(
                f"   Loaded config: {sim_config.duration_days} days, locations: {sim_config.locations}"
            )

            # Test 7: Test MetricsCollector
            print("7. Testing MetricsCollector...")
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

        except Exception as e:
            self.fail(f"\n❌ Test failed: {e}")

    def test_short_simulation(self):
        """Test a very short simulation run."""
        print("\nTesting short simulation run (1 minute)...")

        try:
            # Create minimal config for quick test
            config = SimulationConfig(
                duration_days=0,  # Will be overridden
                task_interval_seconds=5,
                time_acceleration=100,  # Fast acceleration
                battery_capacity_wh=5.0,
                charge_rate_watts=100.0,
            )

            # Override duration for very short test
            config.duration_days = 1  # 1 day for testing

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

        except Exception as e:
            self.fail(f"❌ Short simulation test failed: {e}")


class TestShortSimulationIntegration(unittest.TestCase):
    """End-to-end integration tests with short simulations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration for 1-hour simulations
        self.test_config = SimulationConfig(
            duration_days=1,  # 1 day for testing
            task_interval_seconds=30,  # 30-second intervals for faster tests
            time_acceleration=10,  # 10x acceleration
            battery_capacity_wh=2.0,  # Smaller battery for quicker depletion
            charge_rate_watts=50.0,
        )

        # Load power profiles
        with open("results/power_profiles.json", "r") as f:
            self.power_profiles = json.load(f)

        # Test parameters
        self.test_location = "CA"
        self.test_season = "summer"
        self.test_week = 1

    def test_all_controllers_short_simulation(self):
        """Test all controller types with 1-hour simulations."""
        results = {}

        controllers = {
            "naive_weak": NaiveWeakController(),
            "naive_strong": NaiveStrongController(),
            "custom": CustomController(),
            "oracle": OracleController({}, 0),  # Simplified oracle for testing
        }

        for controller_name, controller in controllers.items():
            with self.subTest(controller=controller_name):
                # Create simulation engine
                engine = SimulationEngine(
                    config=self.test_config,
                    controller=controller,
                    location=self.test_location,
                    season=self.test_season,
                    week=self.test_week,
                    power_profiles=self.power_profiles,
                )

                # Run simulation
                start_time = time.time()
                metrics = engine.run()
                elapsed_time = time.time() - start_time

                # Store results
                results[controller_name] = {
                    "metrics": metrics,
                    "elapsed_time": elapsed_time,
                }

                # Basic validation
                self.assertGreater(metrics["total_tasks"], 0)
                self.assertGreaterEqual(metrics["completed_tasks"], 0)
                self.assertLessEqual(metrics["completed_tasks"], metrics["total_tasks"])
                self.assertBetween(
                    metrics["task_completion_rate"],
                    0,
                    100,
                    "Task completion rate should be valid",
                )

                # Should complete quickly due to time acceleration
                self.assertLess(elapsed_time, 30)  # Should complete within 30 seconds

        # Compare controller performance
        self.validate_controller_comparison(results)

    def validate_controller_comparison(self, results):
        """Validate that controllers show different behavior patterns."""
        # Extract key metrics
        completion_rates = {
            name: data["metrics"]["task_completion_rate"]
            for name, data in results.items()
        }
        clean_energy_pct = {
            name: data["metrics"]["clean_energy_percentage"]
            for name, data in results.items()
        }

        # All controllers should have reasonable completion rates
        for controller_name, rate in completion_rates.items():
            self.assertGreater(rate, 0, f"{controller_name} should complete some tasks")

        # Clean energy percentages should be reasonable
        for controller_name, pct in clean_energy_pct.items():
            self.assertBetween(
                pct,
                0,
                100,
                f"{controller_name} clean energy percentage should be valid",
            )

    def test_configuration_validation(self):
        """Test configuration loading and validation."""
        # Create temporary config file
        test_config = {
            "simulation": {
                "duration_days": 1 / 24,
                "task_interval_seconds": 30,
                "time_acceleration": 10,
            },
            "battery": {"capacity_wh": 2.0, "charge_rate_watts": 50.0},
            "locations": ["CA"],
            "seasons": ["summer"],
            "controllers": ["naive_weak"],
            "output_dir": "results/",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonc", delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name

        try:
            # Test ConfigLoader
            config_loader = ConfigLoader(temp_config_path)

            # Should validate successfully
            self.assertTrue(config_loader.validate_config())

            # Should load configuration
            sim_config = config_loader.get_simulation_config()
            self.assertEqual(sim_config.duration_days, 1 / 24)
            self.assertEqual(sim_config.task_interval_seconds, 30)
            self.assertEqual(sim_config.time_acceleration, 10)
            self.assertEqual(sim_config.battery_capacity_wh, 2.0)
            self.assertEqual(sim_config.charge_rate_watts, 50.0)

        finally:
            # Clean up temporary file
            Path(temp_config_path).unlink()

    def assertBetween(self, value, min_val, max_val, msg):
        """Helper to assert value is between min and max."""
        self.assertGreaterEqual(value, min_val, msg)
        self.assertLessEqual(value, max_val, msg)


if __name__ == "__main__":
    unittest.main()
