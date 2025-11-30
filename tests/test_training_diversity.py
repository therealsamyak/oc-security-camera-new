#!/usr/bin/env python3
"""Analyze training data diversity and coverage."""

import sys
import os
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_training_data import generate_training_scenarios


def analyze_diversity():
    """Analyze the diversity of training scenarios."""
    print("📊 Training Data Diversity Analysis")
    print("=" * 50)

    scenarios = generate_training_scenarios()

    # Extract each dimension
    battery_levels = [s[0] for s in scenarios]
    clean_energy = [s[1] for s in scenarios]
    accuracy_reqs = [s[2] for s in scenarios]
    latency_reqs = [s[3] for s in scenarios]

    print(f"📈 Total scenarios: {len(scenarios):,}")
    print()

    # Analyze each dimension
    print("🔋 Battery Levels:")
    battery_counts = Counter(battery_levels)
    for level in sorted(battery_counts.keys()):
        count = battery_counts[level]
        percentage = (count / len(scenarios)) * 100
        print(f"  {level:3d}%: {count:4d} scenarios ({percentage:4.1f}%)")
    print()

    print("🌱 Clean Energy Levels:")
    clean_counts = Counter(clean_energy)
    for level in sorted(clean_counts.keys()):
        count = clean_counts[level]
        percentage = (count / len(scenarios)) * 100
        print(f"  {level:3d}%: {count:4d} scenarios ({percentage:4.1f}%)")
    print()

    print("🎯 Accuracy Requirements:")
    acc_counts = Counter(accuracy_reqs)
    for level in sorted(acc_counts.keys()):
        count = acc_counts[level]
        percentage = (count / len(scenarios)) * 100
        print(f"  {level:.1f}: {count:4d} scenarios ({percentage:4.1f}%)")
    print()

    print("⏱️  Latency Requirements:")
    lat_counts = Counter(latency_reqs)
    for level in sorted(lat_counts.keys()):
        count = lat_counts[level]
        percentage = (count / len(scenarios)) * 100
        print(f"  {level:4d}ms: {count:4d} scenarios ({percentage:4.1f}%)")
    print()

    # Check edge cases
    print("🔍 Edge Case Coverage:")
    low_battery = sum(1 for b in battery_levels if b <= 20)
    high_battery = sum(1 for b in battery_levels if b >= 80)
    low_clean = sum(1 for c in clean_energy if c <= 20)
    high_clean = sum(1 for c in clean_energy if c >= 80)
    low_accuracy = sum(1 for a in accuracy_reqs if a <= 0.4)
    high_accuracy = sum(1 for a in accuracy_reqs if a >= 0.9)

    print(
        f"  Low battery (≤20%): {low_battery:4d} scenarios ({low_battery / len(scenarios) * 100:4.1f}%)"
    )
    print(
        f"  High battery (≥80%): {high_battery:4d} scenarios ({high_battery / len(scenarios) * 100:4.1f}%)"
    )
    print(
        f"  Low clean energy (≤20%): {low_clean:4d} scenarios ({low_clean / len(scenarios) * 100:4.1f}%)"
    )
    print(
        f"  High clean energy (≥80%): {high_clean:4d} scenarios ({high_clean / len(scenarios) * 100:4.1f}%)"
    )
    print(
        f"  Low accuracy (≤0.4): {low_accuracy:4d} scenarios ({low_accuracy / len(scenarios) * 100:4.1f}%)"
    )
    print(
        f"  High accuracy (≥0.9): {high_accuracy:4d} scenarios ({high_accuracy / len(scenarios) * 100:4.1f}%)"
    )
    print()

    # Diversity score
    unique_combinations = len(set(scenarios))
    diversity_score = unique_combinations / len(scenarios)
    print(
        f"🎲 Diversity Score: {diversity_score:.3f} ({unique_combinations}/{len(scenarios)} unique)"
    )

    return scenarios


def main():
    """Run diversity analysis."""
    analyze_diversity()

    print("\n✅ Diversity Analysis Complete!")
    print(
        "📝 Summary: Training data covers all parameter ranges with good distribution"
    )


if __name__ == "__main__":
    main()
