#!/usr/bin/env python3
"""
CustomController training using gradient descent.
Trains model selection and charging decisions using MIPS-generated training data.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import logging


class CustomController:
    """Custom controller with trained weights for model selection and charging."""

    def __init__(self):
        self.weights = {
            "accuracy_weight": 0.5,  # α
            "latency_weight": 0.3,  # β
            "clean_energy_weight": 0.2,  # γ
        }
        self.model_weights = {}
        self.charge_threshold = 0.5
        self.logger = logging.getLogger(__name__)

    def load_training_data(self, filepath: str) -> List[Dict]:
        """Load training data from JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def extract_features(self, scenario: Dict) -> np.ndarray:
        """Extract features from training scenario."""
        return np.array(
            [
                scenario["battery_level"] / 100.0,
                scenario["clean_energy_percentage"] / 100.0,
                scenario["accuracy_requirement"] / 100.0,
                scenario["latency_requirement"] / 3000.0,
            ]
        )

    def predict_model_and_charge(
        self, features: np.ndarray, available_models: List[str]
    ) -> Tuple[str, bool]:
        """Predict model selection and charging decision using current weights."""
        # Simple linear model for model selection
        model_scores = {}
        for model in available_models:
            if model not in self.model_weights:
                self.model_weights[model] = np.random.random(4)

            score = np.dot(features, self.model_weights[model])
            model_scores[model] = score

        selected_model = max(model_scores.keys(), key=lambda x: model_scores[x])

        # Simple threshold for charging decision
        charge_score = (features[0] < 0.2) * 0.7 + (features[1] > 0.8) * 0.3
        should_charge = charge_score > self.charge_threshold

        return selected_model, should_charge

    def compute_loss(
        self,
        prediction: Tuple[str, bool],
        target: Tuple[str, bool],
        features: np.ndarray,
        available_models: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute loss function: α*(1-accuracy) + β*latency + γ*non_clean_energy"""
        pred_model, pred_charge = prediction
        target_model, target_charge = target

        # Model selection loss
        model_correct = 1.0 if pred_model == target_model else 0.0

        # Charging loss
        charge_correct = 1.0 if pred_charge == target_charge else 0.0

        # Combined accuracy
        total_accuracy = (model_correct + charge_correct) / 2.0

        # Latency penalty (mock based on model)
        latency_penalty = available_models[pred_model]["latency"] / 3000.0

        # Clean energy penalty
        clean_energy_penalty = 0.0
        if target_charge and not pred_charge:
            clean_energy_penalty = 1.0 - features[1]  # Missed clean energy opportunity

        loss = (
            self.weights["accuracy_weight"] * (1 - total_accuracy)
            + self.weights["latency_weight"] * latency_penalty
            + self.weights["clean_energy_weight"] * clean_energy_penalty
        )

        return loss

    def train_step(
        self,
        scenario: Dict,
        available_models: Dict[str, Dict[str, float]],
        learning_rate: float = 0.01,
    ) -> float:
        """Single training step using gradient descent."""
        features = self.extract_features(scenario)
        target_model = scenario["optimal_model"]
        target_charge = scenario["should_charge"]

        # Forward pass
        prediction = self.predict_model_and_charge(
            features, list(available_models.keys())
        )

        # Compute loss
        loss = self.compute_loss(
            prediction, (target_model, target_charge), features, available_models
        )

        # Simple gradient update (mock implementation)
        pred_model, pred_charge = prediction

        if pred_model != target_model:
            # Update model weights
            if pred_model in self.model_weights:
                gradient = learning_rate * (features * (1 if target_charge else -1))
                self.model_weights[pred_model] -= gradient

        if pred_charge != target_charge:
            # Update charge threshold
            if target_charge:
                self.charge_threshold -= learning_rate * 0.1
            else:
                self.charge_threshold += learning_rate * 0.1

            self.charge_threshold = np.clip(self.charge_threshold, 0.0, 1.0)

        return loss

    def train(
        self,
        training_data: List[Dict],
        available_models: Dict[str, Dict[str, float]],
        epochs: int = 100,
        learning_rate: float = 0.01,
    ):
        """Train the CustomController on training data."""
        print(f"Training CustomController for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0.0

            for scenario in training_data:
                loss = self.train_step(scenario, available_models, learning_rate)
                total_loss += loss

            avg_loss = total_loss / len(training_data)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    def save_weights(self, filepath: str):
        """Save trained weights to JSON file."""
        weights_data = {
            "weights": self.weights,
            "model_weights": {k: v.tolist() for k, v in self.model_weights.items()},
            "charge_threshold": self.charge_threshold,
        }

        with open(filepath, "w") as f:
            json.dump(weights_data, f, indent=2)

        print(f"Trained weights saved to {filepath}")

    def load_weights(self, filepath: str):
        """Load trained weights from JSON file."""
        with open(filepath, "r") as f:
            weights_data = json.load(f)

        self.weights = weights_data["weights"]
        self.model_weights = {
            k: np.array(v) for k, v in weights_data["model_weights"].items()
        }
        self.charge_threshold = weights_data["charge_threshold"]

        print(f"Trained weights loaded from {filepath}")


def load_power_profiles() -> Dict[str, Dict[str, float]]:
    """Load power profiles from results."""
    with open("results/power_profiles.json", "r") as f:
        profiles = json.load(f)

    models = {}
    for model_name, data in profiles.items():
        models[model_name] = {
            "accuracy": 85.0 + np.random.uniform(-5, 5),  # Mock accuracy
            "latency": data["avg_inference_time_seconds"] * 1000,  # Convert to ms
            "power_cost": data["model_power_mw"],
        }

    return models


def main():
    """Train CustomController from scratch and save weights."""
    print("Loading training data...")
    try:
        training_data = CustomController().load_training_data(
            "results/training_data.json"
        )
        print(f"Loaded {len(training_data)} training samples")
    except FileNotFoundError:
        print("Training data not found. Please run generate_training_data.py first.")
        return

    print("Loading power profiles...")
    available_models = load_power_profiles()

    print("Initializing CustomController...")
    controller = CustomController()

    print("Starting training...")
    controller.train(training_data, available_models, epochs=100, learning_rate=0.01)

    print("Saving trained weights...")
    controller.save_weights("results/custom_controller_weights.json")

    print("Training complete!")


if __name__ == "__main__":
    main()
