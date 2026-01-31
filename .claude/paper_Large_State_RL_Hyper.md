# LAST-RL: Large-State Reinforcement Learning for Hyper-Heuristics

**Source**: papers/Large_State_RL_for_Hyper_Heuristics.pdf
**Authors**: Lucas Kletzander, Nysret Musliu (TU Wien)
**Venue**: AAAI 2023

## Key Idea

LAST-RL uses **large state representation** (15 features) based on the **trajectory of solution changes** for RL-based hyper-heuristic selection. Key innovations:
1. Rich state features from search history (not just current state)
2. Epsilon-greedy with ILS-inspired probability distribution
3. Solution chains based on Luby sequence

## Core Components

### State Features (15 Total)

**Solution Quality Features (4)**:
```python
f1 = objective_current / objective_best_since_reset
f2 = objective_current / objective_global_best
f3 = objective_best_since_reset / objective_global_best
f4 = (obj_current - obj_global_best) / obj_global_best  # Relative gap
```

**Trajectory Features (6)** - Novel contribution:
```python
# Based on last K=100 solution changes
f5 = fraction_of_improvements_in_last_K
f6 = fraction_of_worsenings_in_last_K
f7 = average_improvement_magnitude
f8 = average_worsening_magnitude
f9 = iterations_since_last_improvement / K
f10 = iterations_since_last_worsening / K
```

**Time Features (3)**:
```python
f11 = elapsed_time / total_time_budget
f12 = average_heuristic_execution_time
f13 = time_since_last_best / total_time
```

**Action History Features (2)**:
```python
f14 = last_action_type  # local_search, mutation, ruin_recreate, crossover
f15 = last_action_improved  # 0 or 1
```

### Learning Method: SARSA(λ) with Tile Coding

**Why Tile Coding?**
- Continuous state → discrete tiles
- Generalization across similar states
- Efficient with linear function approximation

**Q-value approximation**:
```
Q(s, a) = Σ w[tile(s, a, i)] for i in tilings
```

**SARSA(λ) Update**:
```python
def update(s, a, r, s_next, a_next):
    delta = r + gamma * Q(s_next, a_next) - Q(s, a)

    # Eligibility traces for faster learning
    for tile in active_tiles(s, a):
        eligibility[tile] += 1

    for tile in all_tiles:
        w[tile] += alpha * delta * eligibility[tile]
        eligibility[tile] *= gamma * lambda_  # Decay
```

### Epsilon-Greedy with ILS-Inspired Probabilities

Standard epsilon-greedy uses uniform random during exploration. LAST-RL improves this:

```python
def select_action(state):
    if random.random() < epsilon:
        # ILS-inspired exploration
        if last_action_was_local_search:
            # Diversify: prefer perturbation (mutation, ruin-recreate)
            probs = [0.1, 0.35, 0.35, 0.2]  # [LS, MUT, RR, CROSS]
        else:
            # Intensify: prefer local search
            probs = [0.7, 0.1, 0.1, 0.1]
        return np.random.choice(actions, p=probs)
    else:
        # Exploit: choose best Q-value
        return argmax_a(Q(state, a))
```

### Solution Chains (Luby Sequence)

Episode length follows Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, ...

```python
def luby(i):
    """Returns i-th element of Luby sequence."""
    k = 1
    while True:
        if i == 2**k - 1:
            return 2**(k-1)
        elif 2**(k-1) <= i < 2**k - 1:
            return luby(i - 2**(k-1) + 1)
        k += 1

# Episode: apply chain_length heuristics, then reset to best
chain_length = luby(episode_number) * base_multiplier
```

**Novel Adaptations**:
1. Continue chain if not in local search phase (still diversifying)
2. Continue if last heuristic improved (still intensifying)
3. Use average successful chain length to scale Luby values

### Reward Function

Simple improvement-based reward:
```python
def reward(delta_objective, execution_time):
    if delta_objective < 0:  # Improvement (minimization)
        return -delta_objective / execution_time  # Reward efficiency
    else:
        return 0  # No reward for non-improvement
```

## Algorithm Summary

```python
def LAST_RL(instance, timeout):
    # Initialization
    state = initial_state()
    w = zeros(num_tiles)  # Weight vector
    solution = construct_initial()

    while time < timeout:
        chain_length = luby(episode) * scale_factor

        for step in range(chain_length):
            features = extract_features(state, solution)
            action = epsilon_greedy(features, w)

            new_solution, exec_time = apply_heuristic(action, solution)
            delta = objective(new_solution) - objective(solution)
            reward = compute_reward(delta, exec_time)

            next_features = extract_features(next_state, new_solution)
            next_action = epsilon_greedy(next_features, w)

            # SARSA(λ) update
            update_weights(w, features, action, reward, next_features, next_action)

            solution = new_solution if accept(delta) else solution
            state = next_state

            if new_best_found:
                break  # End chain early on improvement

        # Reset to best solution for next chain
        solution = best_since_reset

        if should_restart():
            solution = construct_new_initial()
```

## Results

- **Outperforms**: Previous RL-based hyper-heuristics on HyFlex benchmark
- **State-of-the-art**: On CHeSC 2011 benchmark (6 problem domains)
- **Key insight**: Trajectory features significantly improve performance

## Relevance for Our Project

### Applicable Ideas

1. **Trajectory Features**: Track recent improvements/worsenings
2. **ILS-inspired exploration**: Alternate diversification/intensification
3. **Simple reward**: Just use improvement magnitude

### Simplified Implementation for SCF-PDP

```python
class LargeStateRLSelector:
    """RL selector with trajectory-based state features."""

    def __init__(self, operators, history_size=100):
        self.operators = operators
        self.history_size = history_size
        self.objective_history = deque(maxlen=history_size)
        self.q_table = {}  # Discretized state -> operator -> Q

        # SARSA parameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.lambda_ = 0.5  # Eligibility trace decay

    def get_state_features(self, alns):
        """Extract LAST-RL inspired features."""
        history = list(self.objective_history)

        # Quality features
        f1 = alns.current_objective / alns.best_objective
        f2 = alns.best_objective / alns.initial_objective

        # Trajectory features
        if len(history) > 1:
            improvements = sum(1 for i in range(1, len(history))
                             if history[i] < history[i-1])
            f3 = improvements / (len(history) - 1)
        else:
            f3 = 0.0

        # Time features
        f4 = alns.iterations_without_improvement / 1000

        # Temperature feature
        f5 = alns.temperature / alns.config.initial_temperature

        return [f1, f2, f3, f4, f5]

    def discretize(self, features, bins=5):
        return tuple(min(int(f * bins), bins - 1) for f in features)

    def select(self, state_features):
        state = self.discretize(state_features)

        if random.random() < self.epsilon:
            return random.choice(self.operators)

        if state not in self.q_table:
            self.q_table[state] = {op.name: 0.0 for op in self.operators}

        best_name = max(self.q_table[state], key=self.q_table[state].get)
        return next(op for op in self.operators if op.name == best_name)

    def update(self, state, action, reward, next_state, next_action):
        """SARSA update."""
        s = self.discretize(state)
        s_next = self.discretize(next_state)

        if s not in self.q_table:
            self.q_table[s] = {op.name: 0.0 for op in self.operators}
        if s_next not in self.q_table:
            self.q_table[s_next] = {op.name: 0.0 for op in self.operators}

        q_current = self.q_table[s][action.name]
        q_next = self.q_table[s_next][next_action.name]

        self.q_table[s][action.name] += self.alpha * (
            reward + self.gamma * q_next - q_current
        )
```

## Key Takeaways

1. **Rich state matters**: More features = better operator selection
2. **History is valuable**: Trajectory features capture search dynamics
3. **ILS pattern**: Alternate exploration (perturbation) and exploitation (local search)
4. **Simple rewards work**: Just improvement magnitude, no complex weighting