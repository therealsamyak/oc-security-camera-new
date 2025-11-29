import logging
from typing import Dict, Any

class Battery:
    """Simulates a 4000 mAh battery with charging and discharging."""
    
    def __init__(self, capacity_mah: float = 4000.0, charge_rate_watts: float = 100.0):
        self.capacity_mah = capacity_mah
        self.charge_rate_watts = charge_rate_watts
        self.current_level_mah = capacity_mah  # Start fully charged
        self.logger = logging.getLogger(__name__)
        self.total_energy_used_mwh = 0.0
        self.total_clean_energy_used_mwh = 0.0
    
    def discharge(self, power_mw: float, duration_seconds: float, clean_energy_percentage: float = 0.0) -> bool:
        """
        Discharge battery by specified power for duration.
        
        Args:
            power_mw: Power consumption in milliwatts
            duration_seconds: Duration in seconds
            clean_energy_percentage: Percentage of energy from clean sources (0-100)
            
        Returns:
            True if discharge successful, False if insufficient battery
        """
        energy_mwh = power_mw * (duration_seconds / 3600.0)  # Convert to mWh
        energy_mah = energy_mwh / 3.7  # Convert mWh to mAh (assuming 3.7V Li-ion)
        
        if self.current_level_mah < energy_mah:
            self.logger.error(f"Insufficient battery: {self.current_level_mah:.2f}mAh < {energy_mah:.2f}mAh")
            return False
        
        self.current_level_mah -= energy_mah
        self.total_energy_used_mwh += energy_mwh
        self.total_clean_energy_used_mwh += energy_mwh * (clean_energy_percentage / 100.0)
        return True
    
    def charge(self, duration_seconds: float) -> float:
        """
        Charge battery for specified duration.
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Actual energy added in mAh
        """
        energy_wh = self.charge_rate_watts * (duration_seconds / 3600.0)
        energy_mah = energy_wh / 3.7  # Convert Wh to mAh
        
        space_available = self.capacity_mah - self.current_level_mah
        actual_charge = min(energy_mah, space_available)
        
        self.current_level_mah += actual_charge
        return actual_charge
    
    def get_percentage(self) -> float:
        """Get current battery level as percentage."""
        return (self.current_level_mah / self.capacity_mah) * 100.0
    
    def get_level_mah(self) -> float:
        """Get current battery level in mAh."""
        return self.current_level_mah
    
    def get_total_energy_used_mwh(self) -> float:
        """Get total energy used in mWh."""
        return self.total_energy_used_mwh
    
    def get_total_clean_energy_used_mwh(self) -> float:
        """Get total clean energy used in mWh."""
        return self.total_clean_energy_used_mwh
    
    def reset_energy_tracking(self):
        """Reset energy usage tracking."""
        self.total_energy_used_mwh = 0.0
        self.total_clean_energy_used_mwh = 0.0