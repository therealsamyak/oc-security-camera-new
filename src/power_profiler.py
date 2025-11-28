import psutil
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

class PowerProfiler:
    """Advanced power profiling using psutil for accurate measurement."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.battery_info = self._get_battery_info()
    
    def _get_battery_info(self) -> Dict:
        """Get battery information if available."""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    'has_battery': True,
                    'percent': battery.percent,
                    'power_plugged': battery.power_plugged,
                    'secsleft': battery.secsleft
                }
        except Exception:
            pass
        
        return {'has_battery': False}
    
    def measure_inference_power(self, inference_func, iterations: int = 10) -> Dict[str, float]:
        """
        Measure power consumption during inference.
        
        Args:
            inference_func: Function to run inference
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with power metrics
        """
        # Get baseline measurements
        baseline_cpu = psutil.cpu_percent(interval=1)
        baseline_freq = psutil.cpu_freq()
        baseline_freq_current = baseline_freq.current if baseline_freq else 0
        
        # Monitor system metrics during inference
        cpu_samples = []
        memory_samples = []
        
        start_time = time.time()
        
        for _ in range(iterations):
            # Measure before inference
            cpu_before = psutil.cpu_percent(interval=0.1)
            memory_before = psutil.virtual_memory().percent
            
            # Run inference
            try:
                inference_func()
            except Exception as e:
                self.logger.error(f"Inference failed during profiling: {e}")
                continue
            
            # Measure after inference
            cpu_after = psutil.cpu_percent(interval=0.1)
            memory_after = psutil.virtual_memory().percent
            
            cpu_samples.append(max(cpu_before, cpu_after))
            memory_samples.append(max(memory_before, memory_after))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate averages
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        
        # Estimate power consumption
        # This is a simplified model - real power measurement would need hardware-specific APIs
        estimated_power_watts = self._estimate_power_from_cpu(avg_cpu, baseline_freq_current)
        
        return {
            'duration_seconds': duration,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'estimated_power_watts': estimated_power_watts,
            'estimated_power_mw': estimated_power_watts * 1000,
            'iterations_completed': len(cpu_samples)
        }
    
    def _estimate_power_from_cpu(self, cpu_percent: float, cpu_freq_mhz: float) -> float:
        """
        Estimate power consumption from CPU usage and frequency.
        
        Args:
            cpu_percent: CPU usage percentage
            cpu_freq_mhz: CPU frequency in MHz
            
        Returns:
            Estimated power consumption in watts
        """
        # Simplified power model
        # Base power + dynamic power based on CPU usage and frequency
        base_power_watts = 15.0  # Base system power
        cpu_power_watts = (cpu_percent / 100.0) * (cpu_freq_mhz / 1000.0) * 0.5
        
        return base_power_watts + cpu_power_watts
    
    def get_system_info(self) -> Dict:
        """Get current system information."""
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'cpu_freq_min': cpu_freq.min if cpu_freq else 0,
            'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'battery_info': self.battery_info
        }