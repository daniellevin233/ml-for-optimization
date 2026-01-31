# BALANCE: Bandit-based Adaptive LNS for Multi-Agent Path Finding

**Source**: papers/Balance_Multi_Agent_Path_Finding_for_LNS.pdf
**Authors**: Thomy Phan, Taoan Huang, Bistra Dilkina, Sven Koenig
**Venue**: AAAI 2024

## Key Idea

BALANCE uses a **bi-level MAB scheme** to adapt:
1. **Level 1**: Selection of destroy heuristics
2. **Level 2**: Selection of neighborhood sizes

This addresses two key limitations of vanilla LNS:
- Fixed neighborhood size limits flexibility
- Roulette wheel selection lacks exploration

## Core Algorithm

### Bi-Level MAB Framework

```python
# Level 1: Destroy heuristic selection
arms_heuristic = [H1, H2, H3]  # e.g., Random, Agent-based, Map-based

# Level 2: Neighborhood size selection
arms_size = [N1, N2, N3, N4]  # e.g., [5, 10, 20, 50]

# Joint selection via bi-level MAB
def select(stats_H, stats_N):
    H = mab_select(arms_heuristic, stats_H)  # Select destroy heuristic
    N = mab_select(arms_size, stats_N)       # Select neighborhood size
    return H, N
```

### MAB Algorithms Compared

**1. Roulette Wheel Selection (Baseline)**:
```python
def roulette_select(arms, weights):
    probs = [w / sum(weights) for w in weights]
    return np.random.choice(arms, p=probs)

def roulette_update(arm, reward, weights, reaction_factor=0.1):
    # Exponential smoothing
    weights[arm] = (1 - reaction_factor) * weights[arm] + reaction_factor * reward
```

**2. UCB1 (Upper Confidence Bound)**:
```python
def ucb1_select(arms, counts, total_rewards, total_count):
    ucb_values = []
    for i, arm in enumerate(arms):
        if counts[i] == 0:
            return arm  # Try each arm at least once

        exploitation = total_rewards[i] / counts[i]  # Mean reward
        exploration = np.sqrt(2 * np.log(total_count) / counts[i])
        ucb_values.append(exploitation + exploration)

    return arms[np.argmax(ucb_values)]

def ucb1_update(arm_idx, reward, counts, total_rewards, total_count):
    counts[arm_idx] += 1
    total_rewards[arm_idx] += reward
    total_count += 1
```

**3. Thompson Sampling (Best Performance)**:
```python
def thompson_select(arms, alphas, betas):
    """Sample from Beta distribution for each arm, select highest."""
    samples = [np.random.beta(alphas[i], betas[i]) for i in range(len(arms))]
    return arms[np.argmax(samples)]

def thompson_update(arm_idx, reward, alphas, betas):
    """Update Beta parameters based on binary reward."""
    if reward > 0:  # Success (improvement)
        alphas[arm_idx] += 1
    else:  # Failure (no improvement)
        betas[arm_idx] += 1
```

### Reward Function

Simple cost improvement:
```python
def compute_reward(old_cost, new_cost):
    improvement = old_cost - new_cost
    return max(0, improvement)  # Non-negative reward
```

For binary rewards (Thompson Sampling):
```python
def compute_binary_reward(old_cost, new_cost):
    return 1 if new_cost < old_cost else 0
```

## Complete BALANCE Algorithm

```python
def BALANCE(initial_solution, destroy_heuristics, neighborhood_sizes,
            time_budget, mab_algorithm='thompson'):

    P = initial_solution  # Current solution
    best_cost = cost(P)

    # Initialize MAB statistics
    if mab_algorithm == 'thompson':
        # Beta distribution parameters
        alpha_H = {H: 1.0 for H in destroy_heuristics}
        beta_H = {H: 1.0 for H in destroy_heuristics}
        alpha_N = {N: 1.0 for N in neighborhood_sizes}
        beta_N = {N: 1.0 for N in neighborhood_sizes}
    elif mab_algorithm == 'ucb1':
        counts_H = {H: 0 for H in destroy_heuristics}
        rewards_H = {H: 0.0 for H in destroy_heuristics}
        counts_N = {N: 0 for N in neighborhood_sizes}
        rewards_N = {N: 0.0 for N in neighborhood_sizes}
        total_count = 0

    while time_remaining(time_budget):
        # Level 1: Select destroy heuristic
        if mab_algorithm == 'thompson':
            H = thompson_select(destroy_heuristics, alpha_H, beta_H)
        elif mab_algorithm == 'ucb1':
            H = ucb1_select(destroy_heuristics, counts_H, rewards_H, total_count)

        # Level 2: Select neighborhood size
        if mab_algorithm == 'thompson':
            N = thompson_select(neighborhood_sizes, alpha_N, beta_N)
        elif mab_algorithm == 'ucb1':
            N = ucb1_select(neighborhood_sizes, counts_N, rewards_N, total_count)

        # Destroy and repair
        P_destroyed = H.destroy(P, N)
        P_new = repair(P_destroyed)

        # Compute reward
        old_cost = cost(P)
        new_cost = cost(P_new)
        reward = 1 if new_cost < old_cost else 0

        # Update MAB statistics for both levels
        if mab_algorithm == 'thompson':
            thompson_update(H, reward, alpha_H, beta_H)
            thompson_update(N, reward, alpha_N, beta_N)
        elif mab_algorithm == 'ucb1':
            ucb1_update(H, reward, counts_H, rewards_H, total_count)
            ucb1_update(N, reward, counts_N, rewards_N, total_count)
            total_count += 1

        # Accept improvement
        if new_cost < old_cost:
            P = P_new
            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = P_new

    return best_solution
```

## Results

1. **Thompson Sampling** outperforms UCB1 and roulette wheel
2. **50%+ improvement** over vanilla MAPF-LNS in large-scale scenarios
3. **Adaptive neighborhood size** is crucial for performance
4. **No offline training required** - pure online learning

### Why Thompson Sampling Works Best

- Better exploration in early stages (high uncertainty)
- Naturally balances exploration/exploitation
- Handles non-stationary rewards well (common in LNS)
- Bayesian approach integrates prior knowledge gracefully

## Relevance for Our Project

### Direct Application

The bi-level MAB can be adapted for SCF-PDP:
- **Level 1**: Select destroy operator (Random, WorstCost, LongestRoute)
- **Level 2**: Select removal percentage (10%, 20%, 30%, 40%)

### Implementation for SCF-PDP

```python
class BiLevelMABSelector:
    """Bi-level Thompson Sampling for destroy operator and removal size."""

    def __init__(self, destroy_operators, removal_percentages):
        self.destroy_ops = destroy_operators
        self.removal_pcts = removal_percentages

        # Thompson Sampling: Beta(alpha, beta) for each arm
        self.alpha_destroy = {op.name: 1.0 for op in destroy_operators}
        self.beta_destroy = {op.name: 1.0 for op in destroy_operators}
        self.alpha_removal = {pct: 1.0 for pct in removal_percentages}
        self.beta_removal = {pct: 1.0 for pct in removal_percentages}

    def select(self):
        """Select destroy operator and removal percentage."""
        # Sample for destroy operators
        destroy_samples = {
            name: np.random.beta(self.alpha_destroy[name], self.beta_destroy[name])
            for name in self.alpha_destroy
        }
        best_destroy_name = max(destroy_samples, key=destroy_samples.get)
        destroy_op = next(op for op in self.destroy_ops if op.name == best_destroy_name)

        # Sample for removal percentages
        removal_samples = {
            pct: np.random.beta(self.alpha_removal[pct], self.beta_removal[pct])
            for pct in self.alpha_removal
        }
        removal_pct = max(removal_samples, key=removal_samples.get)

        return destroy_op, removal_pct

    def update(self, destroy_op, removal_pct, improved: bool):
        """Update both levels based on outcome."""
        if improved:
            self.alpha_destroy[destroy_op.name] += 1
            self.alpha_removal[removal_pct] += 1
        else:
            self.beta_destroy[destroy_op.name] += 1
            self.beta_removal[removal_pct] += 1
```

## Key Takeaways

1. **Bi-level selection**: Can optimize multiple aspects simultaneously
2. **Thompson Sampling**: Best MAB algorithm for LNS context
3. **Simple rewards**: Binary (improved/not improved) works well
4. **Online learning**: No need for offline training data
5. **Adaptive parameters**: Neighborhood size should also be adaptive