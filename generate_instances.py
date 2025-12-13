from pathlib import Path


def theoretical_max_weeks(m: int, n: int) -> int:
    """
    Calculate the theoretical maximum number of weeks possible.
    Upper bound: (m*n - 1) / (n - 1)
    This comes from the constraint that each golfer can be paired
    with at most (m*n - 1) other golfers, and in each week they
    meet (n - 1) new golfers in their group.
    """
    total_golfers = m * n
    if n <= 1:
        return 0
    return (total_golfers - 1) // (n - 1)


def is_valid_instance(m: int, n: int, w: int) -> bool:
    """Check if an instance has valid parameters."""
    if m < 2 or n < 2 or w < 1:
        return False
    max_weeks = theoretical_max_weeks(m, n)
    return w <= max_weeks


def generate_instance_file(instances_dir: Path, m: int, n: int, w: int) -> str:
    """Generate an instance file and return the filename."""
    filename = f"sgp_{m}_{n}_{w}.txt"
    filepath = instances_dir / filename

    # Simple format: m n w (space-separated on one line)
    with open(filepath, 'w') as f:
        f.write(f"{m} {n} {w}\n")

    return filename


def generate_all_instances():
    instances_dir = Path("instances")
    instances_dir.mkdir(exist_ok=True)

    generated = set()

    # Required instances: 8-4-x (x=6...10)
    print("Generating required instances (8-4-x)...")
    for w in range(6, 11):
        if is_valid_instance(8, 4, w):
            filename = generate_instance_file(instances_dir, 8, 4, w)
            generated.add((8, 4, w))
            print(f"  Created: {filename}")

    print("\nGenerating diverse instances...")

    # m (groups): 3-8, n (golfers per group): 3-7
    m_values = list(range(3, 10))
    n_values = list(range(3, 8))

    for m in m_values:
        for n in n_values:
            max_w = theoretical_max_weeks(m, n)
            w_values = set()

            # Easy/medium instances only
            w_values.add(3)
            w_values.add(4)

            # Add max weeks
            w_values.add(max_w - 1)
            w_values.add(max_w)

            for w in sorted(w_values):
                if is_valid_instance(m, n, w) and (m, n, w) not in generated:
                    generate_instance_file(instances_dir, m, n, w)
                    generated.add((m, n, w))

    print(f"\n{'='*50}")
    print(f"Total instances generated: {len(generated)}")
    print(f"Instances saved to: {instances_dir.absolute()}")

    # Print statistics
    print(f"\n{'='*50}")
    print("Instance Statistics:")
    print(f"  Groups (m) range: {min(x[0] for x in generated)} - {max(x[0] for x in generated)}")
    print(f"  Golfers per group (n) range: {min(x[1] for x in generated)} - {max(x[1] for x in generated)}")
    print(f"  Weeks (w) range: {min(x[2] for x in generated)} - {max(x[2] for x in generated)}")

    # Categorize by difficulty
    easy = [(m, n, w) for m, n, w in generated if w <= theoretical_max_weeks(m, n) * 0.5]
    medium = [(m, n, w) for m, n, w in generated if theoretical_max_weeks(m, n) * 0.5 < w <= theoretical_max_weeks(m, n) * 0.8]
    hard = [(m, n, w) for m, n, w in generated if w > theoretical_max_weeks(m, n) * 0.8]

    print(f"\n  Easy instances (w <= 50% of max): {len(easy)}")
    print(f"  Medium instances (50% < w <= 80% of max): {len(medium)}")
    print(f"  Hard instances (w > 80% of max): {len(hard)}")

    return generated


if __name__ == "__main__":
    generate_all_instances()
