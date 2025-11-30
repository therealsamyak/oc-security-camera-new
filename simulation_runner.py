#!/usr/bin/env python3
"""
Security Camera Simulation Engine - Phase 4 Implementation

This file orchestrates the complete simulation system for evaluating different
controller strategies under various energy conditions and workloads.

Usage:
    python simulation_runner.py [--parallel] [--workers N] [--config config.jsonc]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation_engine import SimulationEngine, SimulationConfig, TaskGenerator
from controller import NaiveWeakController, NaiveStrongController, OracleController, CustomController
from config_loader import ConfigLoader
from metrics_collector import MetricsCollector, CSVExporter
from logging_config import setup_logging


class SimulationRunner:
    """Orchestrates execution of multiple simulations."""
    
    def __init__(self, config_path: str = "config.jsonc", max_workers: int = 4):
        self.config_loader = ConfigLoader(config_path)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Load power profiles
        self.power_profiles = self._load_power_profiles()
        
        # Initialize exporter
        output_dir = self.config_loader.get_output_dir()
        self.exporter = CSVExporter(output_dir)
        
        # Results storage
        self.all_results = []
        self.failed_simulations = []
        
        # Validate configuration
        if not self.config_loader.validate_config():
            raise ValueError("Invalid configuration")
    
    def _load_power_profiles(self):
        """Load power profiles from results file."""
        import json
        try:
            with open("results/power_profiles.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error("Power profiles file not found")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid power profiles JSON: {e}")
            raise
    
    def _create_controller(self, controller_type: str):
        """Create controller instance based on type."""
        controllers = {
            "naive_weak": NaiveWeakController(),
            "naive_strong": NaiveStrongController(),
            "custom": CustomController(),
        }
        
        if controller_type == "oracle":
            # Oracle controller needs future data - simplified for now
            return OracleController({}, 0)
        
        if controller_type not in controllers:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        return controllers[controller_type]
    
    def _run_single_simulation(self, simulation_params):
        """Run a single simulation and return results."""
        try:
            # Extract parameters
            location = simulation_params["location"]
            season = simulation_params["season"]
            week = simulation_params["week"]
            controller_type = simulation_params["controller"]
            
            self.logger.info(f"Starting simulation: {location} {season} week {week} with {controller_type}")
            
            # Create controller
            controller = self._create_controller(controller_type)
            
            # Get simulation config
            sim_config = self.config_loader.get_simulation_config()
            
            # Create and run simulation
            engine = SimulationEngine(
                config=sim_config,
                controller=controller,
                location=location,
                season=season,
                week=week,
                power_profiles=self.power_profiles
            )
            
            # Run simulation
            metrics = engine.run()
            
            # Add simulation metadata
            from datetime import datetime
            result = {
                **metrics,
                'location': location,
                'season': season,
                'week': week,
                'controller': controller_type,
                'simulation_id': f"{location}_{season}_week{week}_{controller_type}",
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            self.logger.info(f"Completed simulation: {result['simulation_id']}")
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Simulation failed: {simulation_params.get('simulation_id', 'unknown')} - {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Record failure
            from datetime import datetime
            failure_record = {
                **simulation_params,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            self.failed_simulations.append(failure_record)
            
            return None
    
    def _generate_simulation_list(self):
        """Generate list of all simulations to run."""
        simulations = []
        locations = self.config_loader.get_locations()
        seasons = self.config_loader.get_seasons()
        controllers = self.config_loader.get_controllers()
        
        # Generate all combinations: 4 locations × 4 seasons × 4 controllers × 3 weeks = 192
        for location in locations:
            for season in seasons:
                for controller in controllers:
                    for week in range(1, 4):  # 3 weeks per season
                        sim_params = {
                            'location': location,
                            'season': season,
                            'week': week,
                            'controller': controller,
                            'simulation_id': f"{location}_{season}_week{week}_{controller}"
                        }
                        simulations.append(sim_params)
        
        self.logger.info(f"Generated {len(simulations)} simulations to run")
        return simulations
    
    def run_all_simulations(self, parallel: bool = True) -> bool:
        """
        Run all simulations.
        
        Args:
            parallel: Whether to run simulations in parallel
            
        Returns:
            True if all simulations succeeded, False if any failed
        """
        simulations = self._generate_simulation_list()
        successful_results = []
        
        if parallel:
            import concurrent.futures
            # Run simulations in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all simulations
                future_to_sim = {
                    executor.submit(self._run_single_simulation, sim): sim
                    for sim in simulations
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_sim):
                    sim = future_to_sim[future]
                    try:
                        result = future.result()
                        if result:
                            successful_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Simulation {sim['simulation_id']} raised exception: {e}")
                        # Failure already recorded in _run_single_simulation
        else:
            # Run simulations sequentially
            for sim in simulations:
                result = self._run_single_simulation(sim)
                if result:
                    successful_results.append(result)
        
        # Check if any simulations failed
        if len(self.failed_simulations) > 0:
            self.logger.error(f"{len(self.failed_simulations)} simulations failed")
            self.logger.error("Terminating due to failures - no CSV output will be generated")
            return False
        
        # Store successful results
        self.all_results = successful_results
        
        # Export results
        self._export_results()
        
        self.logger.info(f"All {len(successful_results)} simulations completed successfully")
        return True
    
    def _export_results(self):
        """Export simulation results to CSV files."""
        if not self.all_results:
            self.logger.warning("No results to export")
            return
        
        try:
            # Export aggregated results
            aggregated_file = self.exporter.export_aggregated_results(self.all_results)
            if aggregated_file:
                self.logger.info(f"Aggregated results exported to {aggregated_file}")
            
            # Export metadata
            import json
            from datetime import datetime
            metadata = {
                'total_simulations': len(self.all_results),
                'locations': list(set(r['location'] for r in self.all_results)),
                'seasons': list(set(r['season'] for r in self.all_results)),
                'controllers': list(set(r['controller'] for r in self.all_results)),
                'export_timestamp': datetime.now().isoformat(),
                'config_file': str(self.config_loader.config_path)
            }
            
            metadata_file = self.exporter.export_json(metadata, "simulation_metadata.json")
            if metadata_file:
                self.logger.info(f"Metadata exported to {metadata_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise
    
    def get_summary_stats(self):
        """Get summary statistics across all simulations."""
        if not self.all_results:
            return {}
        
        # Calculate aggregate statistics
        total_tasks = sum(r.get('total_tasks', 0) for r in self.all_results)
        total_completed = sum(r.get('completed_tasks', 0) for r in self.all_results)
        total_energy = sum(r.get('total_energy_wh', 0.0) for r in self.all_results)
        total_clean_energy = sum(r.get('clean_energy_wh', 0.0) for r in self.all_results)
        
        # Group by controller
        controller_stats = {}
        for result in self.all_results:
            controller = result.get('controller', 'unknown')
            if controller not in controller_stats:
                controller_stats[controller] = {
                    'count': 0,
                    'avg_completion_rate': 0.0,
                    'avg_clean_energy_pct': 0.0,
                    'total_energy': 0.0
                }
            
            stats = controller_stats[controller]
            stats['count'] += 1
            stats['avg_completion_rate'] += result.get('task_completion_rate', 0.0)
            stats['avg_clean_energy_pct'] += result.get('clean_energy_percentage', 0.0)
            stats['total_energy'] += result.get('total_energy_wh', 0.0)
        
        # Calculate averages
        for controller, stats in controller_stats.items():
            if stats['count'] > 0:
                stats['avg_completion_rate'] /= stats['count']
                stats['avg_clean_energy_pct'] /= stats['count']
        
        return {
            'total_simulations': len(self.all_results),
            'total_tasks': total_tasks,
            'total_completed_tasks': total_completed,
            'overall_completion_rate': (total_completed / total_tasks * 100) if total_tasks > 0 else 0,
            'total_energy_wh': total_energy,
            'total_clean_energy_wh': total_clean_energy,
            'overall_clean_energy_percentage': (total_clean_energy / total_energy * 100) if total_energy > 0 else 0,
            'controller_performance': controller_stats,
            'failed_simulations': len(self.failed_simulations)
        }


def main():
    """Main entry point for simulation runner."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create simulation runner
        runner = SimulationRunner(
            config_path="config.jsonc",
            max_workers=1  # Sequential execution only
        )
        
        # Run simulations sequentially
        success = runner.run_all_simulations(parallel=False)
        
        if success:
            # Print summary statistics
            stats = runner.get_summary_stats()
            logger.info("=== Simulation Summary ===")
            logger.info(f"Total simulations: {stats['total_simulations']}")
            logger.info(f"Overall completion rate: {stats['overall_completion_rate']:.2f}%")
            logger.info(f"Overall clean energy usage: {stats['overall_clean_energy_percentage']:.2f}%")
            logger.info(f"Total energy consumed: {stats['total_energy_wh']:.2f} Wh")
            
            logger.info("\n=== Controller Performance ===")
            for controller, perf in stats['controller_performance'].items():
                logger.info(f"{controller}:")
                logger.info(f"  Simulations: {perf['count']}")
                logger.info(f"  Avg completion rate: {perf['avg_completion_rate']:.2f}%")
                logger.info(f"  Avg clean energy: {perf['avg_clean_energy_pct']:.2f}%")
                logger.info(f"  Total energy: {perf['total_energy']:.2f} Wh")
            
            logger.info("\nAll simulations completed successfully!")
            logger.info(f"Results exported to: {runner.exporter.output_dir}")
            return 0
        else:
            logger.error("Some simulations failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Simulation runner failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())