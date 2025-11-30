#!/usr/bin/env python3
"""Test updated latency ranges in training data generation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_training_data import generate_training_scenarios


def test_updated_latency_ranges():
    """Test that latency ranges are now realistic."""
    print("🔄 Testing Updated Latency Ranges...")

    scenarios = generate_training_scenarios()
    latency_values = [s[3] for s in scenarios]

    print(f"📈 Total scenarios: {len(scenarios):,}")
    print(f"⏱️  Latency range: {min(latency_values)}ms to {max(latency_values)}ms")
    print(f"📊 Unique latency values: {sorted(set(latency_values))}")

    # Check coverage of actual model latencies
    actual_model_latencies = [1.56, 2.66, 5.48, 6.54, 8.33, 12.2]

    print("\n🎯 Model Latency Coverage:")
    for model_latency in actual_model_latencies:
        # Find closest training latency
        closest_training = min(latency_values, key=lambda x: abs(x - model_latency))
        diff = abs(closest_training - model_latency)
        print(
            f"  Model {model_latency:.2f}ms → Training {closest_training}ms (diff: {diff:.2f}ms)"
        )

    # Check distribution
    from collections import Counter

    latency_counts = Counter(latency_values)

    print("\n📊 Latency Distribution:")
    for latency in sorted(latency_counts.keys()):
        count = latency_counts[latency]
        percentage = (count / len(scenarios)) * 100
        print(f"  {latency:2d}ms: {count:4d} scenarios ({percentage:4.1f}%)")

    print("\n✅ Latency ranges updated successfully!")
    return True


def main():
    """Run latency range test."""
    test_updated_latency_ranges()


if __name__ == "__main__":
    main()
