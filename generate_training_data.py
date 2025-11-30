#!/usr/bin/env python3
"""
MIPS solver to generate training data for CustomController.
Generates optimal decisions for diverse scenarios and caches to JSON.
"""

import json
import pulp
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_power_profiles() -> Dict[str, Dict[str, float]]:
    """Load power profiles from results and model data from CSV."""
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
        # Use real accuracy and latency from model-data.csv
        real_data = model_data.get(model_name, {})
        models[model_name] = {
            "accuracy": real_data.get("accuracy", 85.0),  # Fallback to 85% if not found
            "latency": real_data.get(
                "latency", data["avg_inference_time_seconds"] * 1000
            ),  # Use real latency, fallback to power profile
            "power_cost": data[
                "model_power_mw"
            ],  # Keep power data from power profiling
        }

    return models


def solve_mips_scenario(
    battery_level: float,
    clean_energy_percentage: float,
    accuracy_requirement: float,
    latency_requirement: float,
    available_models: Dict[str, Dict[str, float]],
) -> Tuple[str, bool]:
    """
    Solve MIPS for a single scenario to get optimal model and charging decision.
    """
    prob = pulp.LpProblem("Training_Scenario", pulp.LpMaximize)

    model_vars = {
        name: pulp.LpVariable(f"use_{name}", cat="Binary")
        for name in available_models.keys()
    }
    charge_var = pulp.LpVariable("charge", cat="Binary")

    prob += (
        pulp.lpSum(
            [
                available_models[name]["accuracy"] * model_vars[name]
                for name in available_models.keys()
            ]
        )
        - 0.001
        * pulp.lpSum(
            [
                available_models[name]["latency"] * model_vars[name]
                for name in available_models.keys()
            ]
        )
        + 0.01 * clean_energy_percentage * charge_var
    )

    prob += pulp.lpSum(model_vars.values()) == 1

    for name, specs in available_models.items():
        if specs["accuracy"] < accuracy_requirement:
            prob += model_vars[name] == 0
        if specs["latency"] > latency_requirement:
            prob += model_vars[name] == 0

    prob += battery_level + charge_var * 10 <= 100

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_model = list(available_models.keys())[0]
    for name, var in model_vars.items():
        if pulp.value(var) == 1:
            selected_model = name
            break

    should_charge = pulp.value(charge_var) == 1

    return selected_model, should_charge


def generate_training_scenarios(
    seed: Optional[int] = None,
) -> List[Tuple[int, int, float, int]]:
    """Generate purely random training scenarios with optional seed."""
    if seed is not None:
        np.random.seed(seed)

    # Generate 100,000 purely random scenarios
    scenarios = []
    for _ in range(100000):
        battery = np.random.uniform(1, 100)
        clean_energy = np.random.uniform(0, 100)
        acc_req = np.random.uniform(0.2, 1.0)
        lat_req = np.random.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 30])
        scenarios.append((battery, clean_energy, acc_req, lat_req))

    return scenarios


def main():
    """Generate training data and save to JSON."""
    print("Loading power profiles...")
    models = load_power_profiles()

    print("Generating training scenarios...")
    scenarios = generate_training_scenarios()

    print(f"Solving MIPS for {len(scenarios)} scenarios...")
    training_data = []

    for i, (battery, clean_energy, acc_req, lat_req) in enumerate(scenarios):
        if i % 5000 == 0:
            print(f"Progress: {i}/{len(scenarios)}")

        try:
            selected_model, should_charge = solve_mips_scenario(
                battery, clean_energy, acc_req, lat_req, models
            )

            training_data.append(
                {
                    "battery_level": int(battery),
                    "clean_energy_percentage": int(clean_energy),
                    "accuracy_requirement": float(acc_req),
                    "latency_requirement": int(lat_req),
                    "optimal_model": selected_model,
                    "should_charge": bool(should_charge),
                }
            )
        except Exception as e:
            print(f"Error solving scenario {i}: {e}")
            continue

    print(f"Generated {len(training_data)} training samples")

    # Save to JSON
    with open("results/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print("Training data saved to results/training_data.json")


if __name__ == "__main__":
    main()
