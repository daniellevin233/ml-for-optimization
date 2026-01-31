# BALANS: Multi-Armed Bandits-based Adaptive Large Neighborhood Search

**Source**: papers/Multi-Armed_Bandits_Based_ALNS.pdf
**Authors**: Junyang Cai, Serdar Kadıoğlu, Bistra Dilkina
**Venue**: arXiv (2024)

## Key Idea

BALANS is an **online meta-solver** for MIPs that uses Multi-Armed Bandits (MAB) to adaptively select destroy operators in ALNS **without any offline training**. It treats each destroy operator (neighborhood) as an "arm" with unknown reward distribution.

## Core Algorithm

### ALNS(MIP) Framework
1. Start with initial feasible solution from MIP solver
2. Iteratively apply destroy → repair cycle
3. MAB guides destroy operator selection based on online learning
4. Accept/reject via greedy improvement or simulated annealing

### Destroy Operators (8 built-in)
- Crossover, DINS, Local Branching, Mutation
- Proximity Search, Random Objective, RENS, RINS

### MAB Algorithms Studied
1. **ε-greedy**: Explore with probability ε, exploit otherwise
2. **UCB1**: Upper Confidence Bound - balances exploration/exploitation via confidence intervals
3. **Thompson Sampling**: Bayesian approach with Beta distribution priors

## Key MAB Formulation

**Reward Definition** (Critical for LNS):
```
reward = {
    1  if iteration improves objective (new_obj < current_obj)
    0  otherwise
}
```

**UCB1 Selection**:
```
UCB_i = (mean_reward_i) + c * sqrt(2 * ln(total_pulls) / pulls_i)
```
- `c` = exploration coefficient (typically 1.0)
- Balances exploitation (mean reward) with exploration (uncertainty term)

**Thompson Sampling**:
- Maintain Beta(α_i, β_i) for each arm i
- α_i = successes + 1, β_i = failures + 1
- Sample from each Beta, select arm with highest sample

## Implementation Details

### Key Parameters
- **destroy_step**: How many variables to destroy (neighborhood size)
- **incumbent_threshold**: Filter easy instances
- **mip_timeout**: Time limit per repair operation

### Weight Update (for adaptive version)
```python
# After each iteration t:
alpha[i] += reward  # if arm i was selected and got reward
beta[i] += (1 - reward)  # if arm i was selected and got no reward
```

## Results & Insights

1. **Thompson Sampling** performed best overall - better exploration in early stages
2. MAB-based selection outperforms:
   - Single best neighborhood (fixed operator)
   - Adaptive weight update (traditional ALNS)
3. Key insight: **Diversity matters** - even weaker operators contribute when sequenced properly

## Relevance for Our Implementation

### Direct Application to SCF-PDP
- Use MAB to select among our 3 destroy operators
- Binary reward: 1 if solution improves, 0 otherwise
- Thompson Sampling recommended as primary strategy

### Suggested Implementation
```python
class MABOperatorSelector:
    def __init__(self, operators):
        self.operators = operators
        # Thompson Sampling: Beta(alpha, beta) per operator
        self.alpha = {op.name: 1.0 for op in operators}  # prior successes
        self.beta = {op.name: 1.0 for op in operators}   # prior failures

    def select(self):
        # Sample from Beta distribution for each operator
        samples = {
            name: np.random.beta(self.alpha[name], self.beta[name])
            for name in self.alpha
        }
        # Select operator with highest sample
        return max(samples, key=samples.get)

    def update(self, operator_name, improved: bool):
        if improved:
            self.alpha[operator_name] += 1
        else:
            self.beta[operator_name] += 1
```

### Key Differences from Traditional ALNS
| Aspect | Traditional ALNS | MAB-based ALNS |
|--------|-----------------|----------------|
| Selection | Roulette wheel (proportional to weights) | UCB1 or Thompson Sampling |
| Update | Periodic batch update | After every iteration |
| Exploration | Implicit via weights | Explicit via UCB/sampling |
| Training | None needed | None needed (online) |