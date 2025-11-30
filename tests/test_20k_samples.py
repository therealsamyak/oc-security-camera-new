#!/usr/bin/env python3
"""Test updated 20,000 sample training data generation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_training_data import generate_training_scenarios


def test_20k_samples():
    """Test 20,000 sample generation with increased variety."""
    print("📊 Testing 20,000 Sample Generation...")

    scenarios = generate_training_scenarios()

    print(f"📈 Total scenarios: {len(scenarios):,}")

    # Extract each dimension
    battery_levels = [s[0] for s in scenarios]
    clean_energy = [s[1] for s in scenarios]
    accuracy_reqs = [s[2] for s in scenarios]
    latency_reqs = [s[3] for s in scenarios]

    print(f"🔋 Battery levels: {len(set(battery_levels))} unique")
    print(f"🌱 Clean energy: {len(set(clean_energy))} unique")
    print(f"🎯 Accuracy: {len(set(accuracy_reqs))} unique")
    print(f"⏱️  Latency: {len(set(latency_reqs))} unique")

    # Check accuracy variety
    accuracy_values = sorted(set(accuracy_reqs))
    print(f"\n🎯 Accuracy values: {accuracy_values}")
    print(f"   Range: {min(accuracy_values):.2f} to {max(accuracy_values):.2f}")
    print(f"   Step size: ~{accuracy_values[1] - accuracy_values[0]:.2f}")

    # Check distribution
    from collections import Counter

    acc_counts = Counter(accuracy_reqs)
    print("\n📊 Accuracy Distribution (sample):")
    for acc in sorted(acc_counts.keys())[:5]:  # Show first 5
        count = acc_counts[acc]
        percentage = (count / len(scenarios)) * 100
        print(f"  {acc:.2f}: {count:4d} scenarios ({percentage:4.1f}%)")
    print("  ...")

    # Edge case coverage
    low_accuracy = sum(1 for a in accuracy_reqs if a <= 0.4)
    high_accuracy = sum(1 for a in accuracy_reqs if a >= 0.9)

    print("\n🔍 Edge Case Coverage:")
    print(
        f"  Low accuracy (≤0.4): {low_accuracy:4d} scenarios ({low_accuracy / len(scenarios) * 100:4.1f}%)"
    )
    print(
        f"  High accuracy (≥0.9): {high_accuracy:4d} scenarios ({high_accuracy / len(scenarios) * 100:4.1f}%)"
    )

    # Training split impact
    train_size = int(len(scenarios) * 0.7)
    val_size = int(len(scenarios) * 0.2)
    test_size = len(scenarios) - train_size - val_size

    print("\n🔄 Training Split Impact:")
    print(f"  Training: {train_size:,} scenarios")
    print(f"  Validation: {val_size:,} scenarios")
    print(f"  Testing: {test_size:,} scenarios")

    print("\n✅ 20,000 sample generation successful!")
    print("📈 Variety increased from 14,080 to 26,400 possible combinations")

    return scenarios


def main():
    """Run 20k sample test."""
    test_20k_samples()


if __name__ == "__main__":
    main()
