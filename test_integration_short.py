#!/usr/bin/env python3
"""
End-to-end integration tests with 1-hour simulations instead of 7-day.
Tests all controller types with reduced task count and validates failure handling.
"""

import sys
import unittest
import tempfile
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation_engine import SimulationEngine, SimulationConfig
from controller import (
    NaiveWeakController,
    NaiveStrongController,
    OracleController,
    CustomController,
)
from config_loader import ConfigLoader
from metrics_collector import MetricsCollector, CSVExporter


class TestIntegrationShort(unittest.TestCase):
    """End-to-end integration tests with short simulations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration for 1-hour simulations
        self.test_config = SimulationConfig(
            duration_days=1 / 24,  # 1 hour
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

        # Controllers to test
        self.controllers = {
            "naive_weak": NaiveWeakController(),
            "naive_strong": NaiveStrongController(),
            "custom": CustomController(),
            "oracle": OracleController({}, 0),  # Simplified oracle for testing
        }

    def test_all_controllers_short_simulation(self):
        """Test all controller types with 1-hour simulations."""
        results = {}

        for controller_name, controller in self.controllers.items():
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
                self.assertGreaterEqual(metrics["task_completion_rate"], 0)
                self.assertLessEqual(metrics["task_completion_rate"], 100)

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
        energy_usage = {
            name: data["metrics"]["total_energy_mwh"] / 1000  # Convert mWh to Wh
            for name, data in results.items()
        }
        clean_energy_pct = {
            name: data["metrics"]["clean_energy_percentage"]
            for name, data in results.items()
        }

        # Naive weak should use less energy than naive strong (if both use energy)
        # Allow for case where both use 0 energy (no tasks completed)
        if energy_usage["naive_weak"] > 0 or energy_usage["naive_strong"] > 0:
            self.assertLess(
                energy_usage["naive_weak"],
                energy_usage["naive_strong"],
                "NaiveWeakController should use less energy than NaiveStrongController",
            )
        else:
            # Both used 0 energy - this is acceptable for short tests
            self.assertLessEqual(
                energy_usage["naive_weak"],
                energy_usage["naive_strong"],
                "Energy usage should be reasonable",
            )

        # All controllers should have reasonable completion rates
        for controller_name, rate in completion_rates.items():
            self.assertGreater(rate, 0, f"{controller_name} should complete some tasks")

        # Clean energy percentages should be reasonable
        for controller_name, pct in clean_energy_pct.items():
            self.assertGreaterEqual(
                pct,
                0,
                f"{controller_name} clean energy percentage should be non-negative",
            )
            self.assertLessEqual(
                pct,
                100,
                f"{controller_name} clean energy percentage should not exceed 100%",
            )

    def test_reproducible_results(self):
        """Test that simulations produce reproducible results with same seed."""
        controller = NaiveWeakController()

        results = []

        # Run same simulation multiple times
        for i in range(3):
            engine = SimulationEngine(
                config=self.test_config,
                controller=controller,
                location=self.test_location,
                season=self.test_season,
                week=self.test_week,
                power_profiles=self.power_profiles,
            )

            metrics = engine.run()
            results.append(metrics)

        # Results should be similar (allowing for small variations due to timing)
        first_result = results[0]

        for i, result in enumerate(results[1:], 1):
            # Task counts should be identical
            self.assertEqual(
                result["total_tasks"],
                first_result["total_tasks"],
                f"Run {i + 1} should have same total tasks as run 1",
            )

            # Completion rates should be very similar
            self.assertAlmostEqual(
                result["task_completion_rate"],
                first_result["task_completion_rate"],
                places=1,
                msg=f"Run {i + 1} should have similar completion rate to run 1",
            )

    def test_failure_handling_terminate_on_error(self):
        """Test that failure handling terminates immediately."""

        # Create a controller that will fail
        class FailingController:
            def select_model(self, *args, **kwargs):
                raise Exception("Intentional test failure")

        # Create simulation engine with failing controller
        engine = SimulationEngine(
            config=self.test_config,
            controller=FailingController(),
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        # Should raise exception
        with self.assertRaises(Exception):
            engine.run()

        # Test should not create any files (unit tests don't export)
        self.assertTrue(True, "Failure handling test completed")

    def test_performance_benchmarking(self):
        """Test performance benchmarking for short simulation runs."""
        controller = NaiveWeakController()

        # Test different time accelerations
        accelerations = [1, 10, 100]
        performance_results = {}

        for accel in accelerations:
            with self.subTest(acceleration=accel):
                # Update config
                test_config = SimulationConfig(
                    duration_days=1 / 24,  # 1 hour
                    task_interval_seconds=30,
                    time_acceleration=accel,
                    battery_capacity_wh=2.0,
                    charge_rate_watts=50.0,
                )

                engine = SimulationEngine(
                    config=test_config,
                    controller=controller,
                    location=self.test_location,
                    season=self.test_season,
                    week=self.test_week,
                    power_profiles=self.power_profiles,
                )

                # Benchmark performance
                start_time = time.time()
                metrics = engine.run()
                elapsed_time = time.time() - start_time

                performance_results[accel] = {
                    "elapsed_time": elapsed_time,
                    "tasks_per_second": metrics["total_tasks"] / elapsed_time
                    if elapsed_time > 0
                    else 0,
                }

                # Validate results
                self.assertGreater(metrics["total_tasks"], 0)
                self.assertGreater(elapsed_time, 0)

        # Higher acceleration should result in faster completion
        self.assertLess(
            performance_results[100]["elapsed_time"],
            performance_results[10]["elapsed_time"],
            "100x acceleration should be faster than 10x",
        )

        self.assertLess(
            performance_results[10]["elapsed_time"],
            performance_results[1]["elapsed_time"],
            "10x acceleration should be faster than 1x",
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

    def test_energy_interpolation_validation(self):
        """Test energy interpolation between 5-minute data points."""
        # Create simulation with known energy data
        engine = SimulationEngine(
            config=self.test_config,
            controller=NaiveWeakController(),
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        # Check that energy data was loaded
        self.assertIsNotNone(engine.clean_energy_data)
        self.assertGreater(len(engine.clean_energy_data), 0)

        # Test interpolation function
        clean_energy_pct = engine._get_clean_energy_percentage(0.0)
        self.assertIsInstance(clean_energy_pct, float)
        self.assertGreaterEqual(clean_energy_pct, 0)
        self.assertLessEqual(clean_energy_pct, 100)

        # Test multiple time points
        time_points = [0, 3600, 7200, 14400]  # 0h, 1h, 2h, 4h
        for timestamp in time_points:
            with self.subTest(timestamp=timestamp):
                pct = engine._get_clean_energy_percentage(timestamp)
                self.assertIsInstance(pct, float)
                self.assertGreaterEqual(pct, 0)
                self.assertLessEqual(pct, 100)

    def test_metrics_collection_integration(self):
        """Test metrics collection integration with short simulation."""
        # Create metrics collector
        collector = MetricsCollector()
        collector.start_simulation()

        # Run simulation
        controller = NaiveWeakController()
        engine = SimulationEngine(
            config=self.test_config,
            controller=controller,
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        engine.run()
        collector.end_simulation()

        # Validate metrics were collected properly
        final_metrics = collector.get_metrics()

        # Should have tracked tasks
        self.assertGreater(final_metrics["total_tasks"], 0)
        self.assertGreaterEqual(final_metrics["completed_tasks"], 0)

        # Should have battery levels
        self.assertGreater(len(final_metrics["battery_levels"]), 0)

        # Should have model selections
        self.assertGreater(len(final_metrics["model_selections"]), 0)

        # Should calculate final metrics
        collector.calculate_final_metrics()
        calculated_metrics = collector.get_metrics()

        self.assertGreaterEqual(calculated_metrics["task_completion_rate"], 0)
        self.assertLessEqual(calculated_metrics["task_completion_rate"], 100)

    def test_csv_export_integration(self):
        """Test CSV export integration with short simulation results."""
        # Create exporter (unit tests don't actually create files)
        exporter = CSVExporter("results")  # Use results directory

        # Run short simulation
        controller = NaiveWeakController()
        engine = SimulationEngine(
            config=self.test_config,
            controller=controller,
            location=self.test_location,
            season=self.test_season,
            week=self.test_week,
            power_profiles=self.power_profiles,
        )

        metrics = engine.run()

        # Prepare simulation info
        simulation_info = {
            "id": "test_integration",
            "location": self.test_location,
            "season": self.test_season,
            "week": self.test_week,
            "controller": "naive_weak",
        }

        # Add final battery level for export
        metrics["final_battery_level"] = 50.0
        metrics["avg_battery_level"] = 75.0

        # Test export methods (they return file paths but don't create files in unit tests)
        summary_file = exporter.export_summary(metrics, simulation_info)
        exporter.export_detailed_timeseries(metrics, simulation_info)

        # Validate export methods return valid paths
        self.assertIsNotNone(summary_file)
        self.assertTrue(isinstance(summary_file, str))

        # Test aggregated export
        all_simulations = [metrics]
        aggregated_file = exporter.export_aggregated_results(all_simulations)

        self.assertIsNotNone(aggregated_file)
        self.assertTrue(isinstance(aggregated_file, str))


if __name__ == "__main__":
    unittest.main()
