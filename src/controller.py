from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelChoice:
    """Represents a model selection decision."""
    model_name: str
    should_charge: bool
    reasoning: str

class Controller(ABC):
    """Abstract base class for all controllers."""
    
    @abstractmethod
    def select_model(self, 
                    battery_level: float,
                    clean_energy_percentage: float,
                    user_accuracy_requirement: float,
                    user_latency_requirement: float,
                    available_models: Dict[str, Dict[str, float]]) -> ModelChoice:
        """
        Select which model to use based on current conditions.
        
        Args:
            battery_level: Current battery percentage (0-100)
            clean_energy_percentage: Current clean energy percentage (0-100)
            user_accuracy_requirement: Minimum accuracy required (0-100)
            user_latency_requirement: Maximum latency allowed (ms)
            available_models: Dict of model_name -> {accuracy, latency, power_cost}
            
        Returns:
            ModelChoice with selected model and charging decision
        """
        pass