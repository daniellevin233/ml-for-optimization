# Large Neighborhood Search (LNS) - Foundational Reference

**Source**: papers/Pisinger_LNS.pdf
**Authors**: David Pisinger, Stefan Ropke
**Venue**: Handbook of Metaheuristics (2010)

## Overview

This is the foundational paper on LNS and ALNS. It provides the theoretical basis and practical guidelines for implementing large neighborhood search algorithms.

## Key Concepts

### Large Neighborhood Search (LNS)

LNS is a metaheuristic that:
1. Starts with an initial feasible solution
2. Iteratively destroys part of the solution
3. Repairs the destroyed solution
4. Accepts or rejects the new solution

```python
def LNS(initial_solution, max_iterations):
    x = initial_solution
    x_best = x

    for _ in range(max_iterations):
        x_temp = destroy(x)      # Remove part of solution
        x_new = repair(x_temp)    # Reconstruct solution

        if accept(x_new, x):
            x = x_new
            if cost(x_new) < cost(x_best):
                x_best = x_new

    return x_best
```

### Destroy Operators

**Characteristics of good destroy operators**:
1. Remove enough to allow significant changes
2. Preserve structure to guide repair
3. Focus on "bad" parts of solution (for intensification)
4. Random removal (for diversification)

**Common destroy strategies**:
- **Random removal**: Remove q random elements
- **Worst removal**: Remove q worst-cost elements
- **Related removal**: Remove related/clustered elements
- **Historical removal**: Remove elements that were changed before

### Repair Operators

**Common repair strategies**:
- **Greedy insertion**: Insert elements in best position one by one
- **Regret insertion**: Consider second-best options to avoid myopic choices
- **Exact repair**: Use MIP/constraint solver for small neighborhoods

### Acceptance Criteria

1. **Greedy**: Only accept improvements
2. **Simulated Annealing**: Accept worse solutions with probability exp(-Δ/T)
3. **Threshold**: Accept if cost(x_new) < cost(x) + threshold
4. **Record-to-Record**: Accept if cost(x_new) < cost(x_best) + threshold

## Adaptive LNS (ALNS)

ALNS extends LNS by:
1. Maintaining multiple destroy/repair operators
2. Tracking operator performance
3. Adaptively selecting operators based on past success

### Weight Update Mechanism

```python
# After each iteration
if new_best_found:
    score = σ1  # e.g., 33
elif improvement:
    score = σ2  # e.g., 9
elif accepted:
    score = σ3  # e.g., 3
else:
    score = 0

# Accumulate scores over segment
segment_scores[operator] += score
segment_uses[operator] += 1

# Update weights periodically
def update_weights(weights, scores, uses, reaction_factor):
    for op in operators:
        if uses[op] > 0:
            weights[op] = weights[op] * (1 - reaction_factor) + \
                          reaction_factor * (scores[op] / uses[op])
```

### Roulette Wheel Selection

```python
def select_operator(operators, weights):
    total = sum(weights.values())
    probs = [weights[op] / total for op in operators]
    return random.choices(operators, weights=probs)[0]
```

### Key Parameters

| Parameter | Typical Range | Purpose |
|-----------|---------------|---------|
| `reaction_factor` (ρ) | 0.1 - 0.5 | How fast weights adapt |
| `segment_length` | 100 - 500 | Iterations between weight updates |
| `σ1, σ2, σ3` | 33, 9, 3 | Scores for different outcomes |
| `destroy_size` | 10-40% | Fraction of solution to destroy |

## Simulated Annealing in ALNS

```python
def accept_SA(delta, temperature):
    if delta < 0:  # Improvement
        return True
    return random.random() < exp(-delta / temperature)

# Temperature cooling
def cool_temperature(T, cooling_rate):
    return T * cooling_rate  # Geometric cooling
```

**Typical SA parameters**:
- Initial temperature: Set so ~50% of worse solutions accepted initially
- Cooling rate: 0.99 - 0.9999 (depends on iteration count)

## Theoretical Insights

### Why Large Neighborhoods Work
- Larger neighborhoods have better local optima
- Can escape from local minima without explicit diversification
- More likely to find globally good solutions

### Destroy Size Trade-off
- **Small neighborhoods**: Fast but limited improvement
- **Large neighborhoods**: Slower but better exploration
- **Adaptive size**: Best of both worlds (see BALANCE paper)

## Algorithm Template

```python
class ALNS:
    def __init__(self, destroy_ops, repair_ops, config):
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.weights_d = {op: 1.0 for op in destroy_ops}
        self.weights_r = {op: 1.0 for op in repair_ops}
        self.config = config

    def run(self, initial_solution):
        x = initial_solution
        x_best = x.copy()
        T = self.config.initial_temperature

        # Tracking for weight updates
        scores_d = {op: 0 for op in self.destroy_ops}
        uses_d = {op: 0 for op in self.destroy_ops}
        scores_r = {op: 0 for op in self.repair_ops}
        uses_r = {op: 0 for op in self.repair_ops}

        for iteration in range(self.config.max_iterations):
            # Select operators
            d_op = self.roulette_select(self.destroy_ops, self.weights_d)
            r_op = self.roulette_select(self.repair_ops, self.weights_r)

            # Apply destroy-repair
            x_temp = d_op.apply(x)
            x_new = r_op.apply(x_temp)

            # Evaluate
            delta = cost(x_new) - cost(x)
            score = self.evaluate_score(x_new, x, x_best)

            # Update tracking
            scores_d[d_op] += score
            uses_d[d_op] += 1
            scores_r[r_op] += score
            uses_r[r_op] += 1

            # Accept/reject
            if self.accept(delta, T):
                x = x_new
                if cost(x_new) < cost(x_best):
                    x_best = x_new.copy()

            # Periodic weight update
            if (iteration + 1) % self.config.segment_length == 0:
                self.update_weights(self.weights_d, scores_d, uses_d)
                self.update_weights(self.weights_r, scores_r, uses_r)
                # Reset tracking
                scores_d = {op: 0 for op in self.destroy_ops}
                uses_d = {op: 0 for op in self.destroy_ops}
                scores_r = {op: 0 for op in self.repair_ops}
                uses_r = {op: 0 for op in self.repair_ops}

            # Cool temperature
            T *= self.config.cooling_rate

        return x_best
```

## Key Takeaways

1. **ALNS is flexible**: Can incorporate any destroy/repair heuristics
2. **Adaptation is key**: Weights should reflect recent operator success
3. **Balance exploration/exploitation**: Through acceptance criterion and operator diversity
4. **Segment-based updates**: More stable than per-iteration updates
5. **Simple scoring**: Just track if improvement/accepted/rejected

This foundational work establishes the baseline that MAB and RL approaches aim to improve upon.