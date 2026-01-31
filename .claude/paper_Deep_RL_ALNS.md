# DR-ALNS: Deep Reinforcement Learning for ALNS Control

**Source**: papers/Deep_RL_for_ALNS.pdf
**Authors**: Robbert Reijnen, Yingqian Zhang, Hoong Chuin Lau, Zaharah Bukhsh
**Venue**: ICAPS 2024

## Key Idea

DR-ALNS integrates **Deep Reinforcement Learning (DRL)** into ALNS to learn:
1. Which destroy/repair operators to select
2. How to configure operator parameters (destroy severity)
3. How to control acceptance criterion (SA temperature)

The approach is **problem-agnostic** - only needs to adapt action space for new problems.

## Core Architecture

### MDP Formulation

**State Space** (14 features, problem-independent):
```python
state = [
    objective_current / objective_initial,  # Normalized current objective
    objective_best / objective_initial,     # Normalized best objective
    objective_current / objective_best,     # Relative quality
    iterations_since_improvement / max_iter,
    temperature / initial_temperature,
    acceptance_rate_recent,                 # Recent acceptance rate
    improvement_rate_recent,                # Recent improvement rate
    # ... destroy operator performance metrics
    # ... repair operator performance metrics
]
```

**Action Space**:
```python
action = {
    'destroy_operator': Discrete(num_destroy_ops),
    'repair_operator': Discrete(num_repair_ops),
    'destroy_severity': Discrete([0.1, 0.2, 0.3, 0.4]),  # % to destroy
    'temperature_multiplier': Discrete([0.5, 1.0, 1.5, 2.0])
}
```

**Reward Function**:
```python
def compute_reward(old_best, new_best, old_current, new_current):
    if new_best < old_best:
        return 10.0  # New global best
    elif new_current < old_current:
        return 1.0   # Improvement (not best)
    else:
        return -0.1  # No improvement (small penalty)
```

### Neural Network Architecture

```
State (14 features)
    ↓
Dense(128, ReLU)
    ↓
Dense(64, ReLU)
    ↓
Output heads:
  - Destroy operator logits
  - Repair operator logits
  - Destroy severity logits
  - Temperature multiplier logits
```

### Training Algorithm

Uses **PPO (Proximal Policy Optimization)**:
1. Collect trajectories by running ALNS with current policy
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Update policy with clipped surrogate objective

## Key Design Decisions

### State Features (Problem-Agnostic)
- **Normalized objectives**: Allows transfer across instance sizes
- **Relative metrics**: Independent of absolute scale
- **Search dynamics**: Acceptance rate, improvement rate, stagnation
- **Operator history**: Recent performance of each operator

### Why DRL Over MAB?
1. **Richer state representation**: MAB only considers arm statistics
2. **Parameter control**: DRL can also tune destroy size, temperature
3. **Pattern learning**: Can learn when certain operators work better

### Training Efficiency
- Much fewer samples than end-to-end DRL approaches
- Policy learned on small instances generalizes to larger ones
- Single policy works across problem variants (TSP, CVRP, mTSP)

## Implementation Guide

### Simplified Q-Learning Version (for our project)
```python
class QLearningOperatorSelector:
    def __init__(self, operators, state_bins=10, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.operators = operators
        self.n_ops = len(operators)
        self.state_bins = state_bins
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Q-table: discretized state -> operator -> Q-value
        self.q_table = {}

    def discretize_state(self, state_features):
        """Bin continuous state features into discrete state."""
        return tuple(
            min(int(f * self.state_bins), self.state_bins - 1)
            for f in state_features
        )

    def get_state_features(self, alns):
        """Extract problem-agnostic state features."""
        return [
            alns.current_objective / alns.initial_objective,
            alns.best_objective / alns.initial_objective,
            alns.temperature / alns.config.initial_temperature,
            min(alns.iterations_without_improvement / 1000, 1.0),
        ]

    def select(self, state):
        discrete_state = self.discretize_state(state)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(self.operators)

        # Exploit: choose best Q-value
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = {op.name: 0.0 for op in self.operators}

        q_values = self.q_table[discrete_state]
        best_op_name = max(q_values, key=q_values.get)
        return next(op for op in self.operators if op.name == best_op_name)

    def update(self, state, action, reward, next_state):
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)

        if s not in self.q_table:
            self.q_table[s] = {op.name: 0.0 for op in self.operators}
        if s_next not in self.q_table:
            self.q_table[s_next] = {op.name: 0.0 for op in self.operators}

        # Q-learning update
        max_next_q = max(self.q_table[s_next].values())
        self.q_table[s][action.name] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[s][action.name]
        )
```

### DQN Version (More Sophisticated)
```python
class DQNOperatorSelector:
    def __init__(self, state_dim, n_operators, hidden_dim=64):
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_operators)
        )
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)

    def select(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_operators - 1)

        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state))
            return q_values.argmax().item()

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Standard DQN loss computation
        # ...
```

## Results & Insights

1. **Outperforms**: Vanilla ALNS, Bayesian-tuned ALNS, competition winners
2. **Generalizes**: Policy trained on size 20 works on size 100
3. **Transfers**: TSP policy works for CVRP and mTSP without retraining

## Key Takeaways for Our Project

### Recommended Approach
1. **Simple Start**: Q-learning with discretized state (4-5 features)
2. **State Features**: Use normalized objectives, temperature, stagnation
3. **Reward**: +10 for new best, +1 for accepted improvement, 0 otherwise
4. **Training**: Online during ALNS execution (no separate training phase)

### Comparison with MAB
| Aspect | MAB | Q-Learning/DRL |
|--------|-----|----------------|
| State | Arm statistics only | Rich feature vector |
| Learning | Online, simple | Online or offline |
| Complexity | Low | Medium-High |
| Parameters | Few | More to tune |
| Generalization | Per-instance | Can transfer |