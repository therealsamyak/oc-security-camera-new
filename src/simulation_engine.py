import logging
import random
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from battery import Battery
from controller import Controller
from energy_data import EnergyData


@dataclass
class Task:
    """Represents a single inference task."""

    timestamp: float
    accuracy_requirement: float
    latency_requirement: float
    completed: bool = False
    model_used: Optional[str] = None
    energy_used_mwh: float = 0.0
    clean_energy_used_mwh: float = 0.0
    missed_deadline: bool = False


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""

    duration_days: int = 7
    task_interval_seconds: int = 5
    time_acceleration: int = 1
    battery_capacity_wh: float = 5.0
    charge_rate_watts: float = 100.0
    locations: Optional[List[str]] = None
    seasons: Optional[List[str]] = None

    def __post_init__(self):
        if self.locations is None:
            self.locations = ["CA", "FL", "NW", "NY"]
        if self.seasons is None:
            self.seasons = ["winter", "spring", "summer", "fall"]


class TaskGenerator:
    """Generates realistic security camera workload."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_task(self, timestamp: float) -> Optional[Task]:
        """Generate a single task with random requirements."""
        # Security cameras have periodic activity with some randomness
        if self.rng.random() < 0.1:  # 10% chance of no task
            return None

        accuracy_req = self.rng.uniform(70.0, 95.0)
        latency_req = self.rng.uniform(1000.0, 3000.0)

        return Task(
            timestamp=timestamp,
            accuracy_requirement=accuracy_req,
            latency_requirement=latency_req,
        )


class SimulationEngine:
    """Core simulation engine for security camera operations."""

    def __init__(
        self,
        config: SimulationConfig,
        controller: Controller,
        location: str,
        season: str,
        week: int,
        power_profiles: Dict,
    ):
        self.config = config
        self.controller = controller
        self.location = location
        self.season = season
        self.week = week
        self.power_profiles = power_profiles

        # Initialize components
        self.battery = Battery(
            capacity_wh=config.battery_capacity_wh,
            charge_rate_watts=config.charge_rate_watts,
        )

        self.energy_data = EnergyData()
        self.task_generator = TaskGenerator()
        self.logger = logging.getLogger(__name__)

        # Simulation state
        self.current_time = 0.0
        self.tasks: List[Task] = []
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "missed_deadlines": 0,
            "small_model_tasks": 0,
            "large_model_tasks": 0,
            "small_model_misses": 0,
            "large_model_misses": 0,
            "total_energy_mwh": 0.0,
            "clean_energy_mwh": 0.0,
            "battery_levels": [],
            "model_selections": {model: 0 for model in power_profiles.keys()},
        }

        # Load energy data for location and season
        self.clean_energy_data = self._load_clean_energy_data()

    def _load_clean_energy_data(self) -> Dict[int, float]:
        """Load clean energy data for specific location and season."""
        try:
            # Map location to filename
            location_files = {
                "CA": "US-CAL-LDWP_2024_5_minute.csv",
                "FL": "US-FLA-FPL_2024_5_minute.csv",
                "NW": "US-NW-PSEI_2024_5_minute.csv",
                "NY": "US-NY-NYIS_2024_5_minute.csv",
            }

            filename = location_files.get(self.location)
            if not filename:
                raise ValueError(f"Unknown location: {self.location}")

            # Load energy data using existing EnergyData class
            region_name = filename.replace(".csv", "")
            if region_name not in self.energy_data.data:
                raise ValueError(f"Region {region_name} not found in energy data")
            data = self.energy_data.data[region_name].to_dict("records")

            # Filter by season and interpolate to 5-second intervals
            season_data = self._filter_by_season(data, self.season)
            return self._interpolate_energy_data(season_data)

        except Exception as e:
            self.logger.error(f"Failed to load energy data: {e}")
            # Fallback to default clean energy profile
            return self._create_default_energy_profile()

    def _filter_by_season(self, data: List[Dict], season: str) -> List[Dict]:
        """Filter energy data by season."""
        # Simple season mapping - would need more sophisticated logic for real data
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11],
        }

        months = season_months.get(season, [1, 2, 3])
        filtered = []

        for entry in data:
            try:
                month = datetime.fromisoformat(entry["timestamp"]).month
                if month in months:
                    filtered.append(entry)
            except (KeyError, ValueError):
                continue

        return filtered

    def _interpolate_energy_data(self, data: List[Dict]) -> Dict[int, float]:
        """Interpolate 5-minute energy data to 5-second intervals."""
        if not data:
            return self._create_default_energy_profile()

        # Create mapping from timestamp (seconds) to clean energy percentage
        energy_map = {}

        # Sort data by timestamp
        data.sort(key=lambda x: x["timestamp"])

        # Convert to seconds and interpolate
        for i in range(len(data) - 1):
            current_time = datetime.fromisoformat(data[i]["timestamp"])
            next_time = datetime.fromisoformat(data[i + 1]["timestamp"])

            current_seconds = (
                current_time - current_time.replace(hour=0, minute=0, second=0)
            ).total_seconds()
            next_seconds = (
                next_time - next_time.replace(hour=0, minute=0, second=0)
            ).total_seconds()

            current_clean = data[i].get("clean_energy_percentage", 50.0)
            next_clean = data[i + 1].get("clean_energy_percentage", 50.0)

            # Interpolate for each 5-second interval
            steps = int((next_seconds - current_seconds) / 5)
            for step in range(steps):
                t = step / steps
                interpolated_clean = current_clean + t * (next_clean - current_clean)
                timestamp = current_seconds + step * 5
                energy_map[int(timestamp)] = interpolated_clean

        return energy_map

    def _create_default_energy_profile(self) -> Dict[int, float]:
        """Create default clean energy profile when data loading fails."""
        # Simple sinusoidal pattern for demonstration
        energy_map = {}
        for seconds in range(0, 24 * 3600, 5):  # 24 hours in 5-second steps
            hour = seconds / 3600
            # Peak solar at noon, lowest at night
            clean_percentage = max(0, 50 + 40 * ((hour - 12) / 12) ** 2)
            energy_map[seconds] = clean_percentage
        return energy_map

    def _get_clean_energy_percentage(self, timestamp: float) -> float:
        """Get clean energy percentage for given timestamp."""
        # Convert timestamp to seconds since midnight
        seconds_in_day = int(timestamp % (24 * 3600))

        # Find closest timestamp in energy data
        available_times = sorted(self.clean_energy_data.keys())
        if not available_times:
            return 50.0  # Default fallback

        # Find the closest time point
        closest_time = min(available_times, key=lambda x: abs(x - seconds_in_day))
        return self.clean_energy_data.get(closest_time, 50.0)

    def _get_available_models(self) -> Dict[str, Dict[str, float]]:
        """Get available models with their specs."""
        models = {}
        for name, profile in self.power_profiles.items():
            models[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"]
                * 1000,  # Convert to ms
                "power_cost": profile["model_power_mw"],  # Power in mW
            }
        return models

    def _execute_task(self, task: Task) -> bool:
        """Execute a single task and return success status."""
        battery_level = self.battery.get_percentage()
        clean_energy_pct = self._get_clean_energy_percentage(task.timestamp)
        available_models = self._get_available_models()

        # Get controller decision
        choice = self.controller.select_model(
            battery_level=battery_level,
            clean_energy_percentage=clean_energy_pct,
            user_accuracy_requirement=task.accuracy_requirement,
            user_latency_requirement=task.latency_requirement,
            available_models=available_models,
        )

        # Check if selected model meets requirements
        model_specs = available_models[choice.model_name]
        meets_accuracy = model_specs["accuracy"] >= task.accuracy_requirement
        meets_latency = model_specs["latency"] <= task.latency_requirement

        if not (meets_accuracy and meets_latency):
            task.missed_deadline = True
            self.metrics["missed_deadlines"] += 1

            # Track model-specific misses
            if choice.model_name in ["YOLOv10_N", "YOLOv10_S"]:
                self.metrics["small_model_misses"] += 1
            else:
                self.metrics["large_model_misses"] += 1

            return False

        # Execute inference
        power_mw = model_specs["power_cost"]
        duration_seconds = model_specs["latency"] / 1000  # Convert ms to seconds

        # Try to discharge battery
        success = self.battery.discharge(
            power_mw=power_mw,
            duration_seconds=duration_seconds,
            clean_energy_percentage=clean_energy_pct,
        )

        if not success:
            task.missed_deadline = True
            self.metrics["missed_deadlines"] += 1
            return False

        # Update task and metrics
        task.completed = True
        task.model_used = choice.model_name
        task.energy_used_mwh = power_mw * (duration_seconds / 3600)
        task.clean_energy_used_mwh = task.energy_used_mwh * (clean_energy_pct / 100)

        self.metrics["completed_tasks"] += 1
        self.metrics["total_energy_mwh"] += task.energy_used_mwh
        self.metrics["clean_energy_mwh"] += task.clean_energy_used_mwh
        self.metrics["model_selections"][choice.model_name] += 1

        # Track model usage categories
        if choice.model_name in ["YOLOv10_N", "YOLOv10_S"]:
            self.metrics["small_model_tasks"] += 1
        else:
            self.metrics["large_model_tasks"] += 1

        # Handle charging decision
        if choice.should_charge:
            self.battery.charge(self.config.task_interval_seconds)

        return True

    def run(self) -> Dict:
        """Run the complete simulation."""
        self.logger.info(
            f"Starting simulation: {self.location} {self.season} week {self.week}"
        )

        duration_seconds = self.config.duration_days * 24 * 3600
        task_interval = self.config.task_interval_seconds

        start_time = time.time()

        for timestamp in range(0, int(duration_seconds), task_interval):
            self.current_time = timestamp

            # Generate task
            task = self.task_generator.generate_task(timestamp)
            if task is not None:
                self.tasks.append(task)
                self.metrics["total_tasks"] += 1
                self._execute_task(task)

            # Track battery level
            self.metrics["battery_levels"].append(
                {"timestamp": timestamp, "level": self.battery.get_percentage()}
            )

            # Apply time acceleration
            if self.config.time_acceleration > 1:
                time.sleep(
                    0.001 / self.config.time_acceleration
                )  # Minimal delay for acceleration

        elapsed = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed:.2f} seconds")

        # Calculate final metrics
        self._calculate_final_metrics()

        return self.metrics

    def _calculate_final_metrics(self):
        """Calculate final simulation metrics."""
        if self.metrics["total_tasks"] == 0:
            return

        # Calculate miss rates
        if self.metrics["small_model_tasks"] > 0:
            self.metrics["small_model_miss_rate"] = (
                self.metrics["small_model_misses"]
                / self.metrics["small_model_tasks"]
                * 100
            )
        else:
            self.metrics["small_model_miss_rate"] = 0.0

        if self.metrics["large_model_tasks"] > 0:
            self.metrics["large_model_miss_rate"] = (
                self.metrics["large_model_misses"]
                / self.metrics["large_model_tasks"]
                * 100
            )
        else:
            self.metrics["large_model_miss_rate"] = 0.0

        # Calculate clean energy percentage
        if self.metrics["total_energy_mwh"] > 0:
            self.metrics["clean_energy_percentage"] = (
                self.metrics["clean_energy_mwh"]
                / self.metrics["total_energy_mwh"]
                * 100
            )
        else:
            self.metrics["clean_energy_percentage"] = 0.0

        # Calculate task completion rate
        self.metrics["task_completion_rate"] = (
            self.metrics["completed_tasks"] / self.metrics["total_tasks"] * 100
        )
