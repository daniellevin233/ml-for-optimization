# Design Proposal: MAB-Based Operator Selection for ALNS

## Executive Summary

This document outlines the design for extending the existing ALNS implementation with Multi-Armed Bandit (MAB) based operator selection strategies. The goal is to evaluate whether MAB approaches (UCB1, Thompson Sampling) can outperform the current adaptive weight mechanism.

**Key Design Principle**: Extract operator selection into a pluggable strategy interface with minimal changes to the core ALNS algorithm.

---

## Problem Context

### Current ALNS Implementation

The existing ALNS (`src/alns/alns.py`) uses **adaptive weights** for operator selection:
- Maintains weights for each destroy/repair operator
- Selects operators probabilistically (roulette wheel)
- Updates weights every 100 iterations based on accumulated scores
- Uses reaction factor (γ = 0.1) to control adaptation speed

### Limitations of Adaptive Weights

1. **Slow adaptation**: Updates occur periodically, not after each iteration
2. **No exploration bonus**: Selection purely based on past average performance
3. **Ignores uncertainty**: Treats well-tested and under-tested operators equally

### MAB Advantages (Hypothesized)

1. **Explicit exploration**: UCB1 adds exploration bonus; Thompson Sampling samples from posterior
2. **Immediate adaptation**: Updates occur after each iteration
3. **Uncertainty-aware**: Naturally explores under-tested operators more

---

## Architecture Decision: Strategy Pattern

### Why Strategy Pattern?

We chose **composition over inheritance** for operator selection:

**Advantages**:
- Selection logic orthogonal to ALNS main loop
- Easy to swap selectors without duplicating ALNS code
- Better testability (can unit test selectors independently)
- Can use different selectors for destroy vs. repair operators

**Alternative Considered**: Subclassing ALNS (e.g., `MAB_ALNS`, `UCB1_ALNS`)
- **Rejected**: Would duplicate entire ALNS loop for minor selection logic changes

### Core Interface: OperatorSelector

All selection strategies implement this interface:
```python
class OperatorSelector(ABC):
    def select(state) -> Operator         # Choose next operator
    def update(operator, reward)          # Learn from outcome
    def get_statistics() -> dict          # For analysis
```

This allows **dependency injection** into ALNS constructor.

---

## Selection Strategies

### 1. Adaptive Weights (Baseline)

**Mechanism**:
- Maintains weight ρᵢ for each operator i
- Selects via roulette wheel: P(i) = ρᵢ / Σρⱼ
- Updates periodically: ρᵢ ← ρᵢ·(1-γ) + γ·(sᵢ/aᵢ)
  - sᵢ: accumulated score in period
  - aᵢ: number of applications in period
  - γ: reaction factor (0.1)

**Characteristics**:
- **Exploration**: Minimal (depends only on noise from probabilistic selection)
- **Exploitation**: Strong (converges to best operator)
- **Adaptation speed**: Slow (periodic updates)

### 2. UCB1 (Upper Confidence Bound)

**Mechanism**:
- Selects operator with highest UCB value: UCB(i) = μᵢ + c√(2ln(t)/nᵢ)
  - μᵢ: average reward of operator i
  - nᵢ: number of times operator i selected
  - t: total selections
  - c: exploration parameter (typically 1.0)

**Characteristics**:
- **Exploration**: Explicit via confidence interval
- **Exploitation**: Balances with exploration automatically
- **Adaptation speed**: Fast (updates after each iteration)

**Tuning parameter**: Exploration constant c
- Higher c: More exploration
- Lower c: More exploitation
- Default: c = 1.0 (theoretical optimal for stationary bandits)

### 3. Thompson Sampling

**Mechanism**:
- Maintains Beta(αᵢ, βᵢ) distribution for each operator i
- Samples θᵢ ~ Beta(αᵢ, βᵢ) for each operator
- Selects operator with highest sample: argmax θᵢ
- Updates: αᵢ += 1 if reward > 0, else βᵢ += 1

**Characteristics**:
- **Exploration**: Probabilistic sampling from posterior
- **Exploitation**: Naturally focuses on high-reward operators
- **Adaptation speed**: Fast (updates after each iteration)

**Advantage over UCB1**: Better in non-stationary environments (operator effectiveness changes over search)

---

## Reward Function Design

### Options Considered

**Option A: Binary (0/1)**
- reward = 1 if improvement, else 0
- Simple for Thompson Sampling (Beta-Bernoulli)
- Ignores magnitude of improvement

**Option B: Continuous (Δ objective)**
- reward = old_obj - new_obj
- Captures improvement magnitude
- Requires different Thompson Sampling (Normal-Gamma prior)

**Option C: Tiered (10/1/0)** ✓ **CHOSEN**
- reward = 10 if new global best
- reward = 1 if accepted improvement
- reward = 0 otherwise
- **Rationale**: Matches baseline scoring system; emphasizes finding new best solutions

### Conversion for Thompson Sampling

Thompson Sampling requires binary rewards for Beta-Bernoulli:
```
reward_binary = 1 if reward > 0 else 0
```

This treats "any improvement" as success.

---

## Operator Selection Scope

### Decision: Select Both Destroy AND Repair

**Alternatives Considered**:

1. **Destroy only** (like BALANS paper)
   - Papers use single "exact" repair (MIP solver)
   - We have 3 diverse repair operators
   - Would waste existing diversity

2. **Destroy + Repair pairs** (9 arms total)
   - Captures synergies
   - Too many arms = slower learning (9 vs. 6)
   - Harder to interpret results

3. **Independent selection** ✓ **CHOSEN**
   - Two independent selectors: one for destroy, one for repair
   - Each selector has 3 arms
   - **Rationale**: Explores full 3×3 space with only 6 total arms; matches baseline behavior

---

## Experimental Design

### Research Questions (Focused)

**RQ1**: Does MAB outperform adaptive weights?
- Metric: Final objective value
- Analysis: Wilcoxon signed-rank test

**RQ2**: Which MAB algorithm is best?
- Compare: Adaptive Weights vs. UCB1 vs. Thompson Sampling

**RQ4**: How does selection evolve?
- Track: Operator usage over time
- Show: Exploration → exploitation transition

### Algorithms to Compare

1. **AdaptiveWeights** - Baseline
2. **UCB1** - Exploration via confidence bounds
3. **ThompsonSampling** - Probabilistic exploration

Only 3 algorithms to keep scope manageable.

### Evaluation Metrics

**Primary**:
- Final objective value (median, mean, std dev)
- Statistical significance (Wilcoxon test p-values)

**Secondary**:
- Convergence speed (iteration to best solution)
- Operator selection frequencies (overall and over time)
- Average reward per operator

### Instance Selection

- **Sizes**: 50, 100, 200, 500 requests
- **Count**: 30 instances per size (minimum for statistical power)
- **Rationale**: 1000+ requests too slow for extensive experimentation

---

## Deliverables for Report

### Plots (Using Existing Framework)

**Plot 1: Convergence Comparison**
- X: Iteration, Y: Best objective
- Lines: One per algorithm
- Purpose: Show which converges faster

**Plot 2: Solution Quality Distribution**
- Boxplot: Algorithm vs. final objective
- Purpose: Statistical comparison

**Plot 3: Operator Selection Evolution**
- Stacked area or grouped bar chart
- X: Time segments, Y: Selection frequency
- Separate for destroy and repair
- Purpose: Visualize exploration → exploitation

### Tables (LaTeX for Report)

**Table 1: Algorithm Comparison**
- Columns: Algorithm, Median, Mean, Std Dev, p-value
- Shows: Statistical significance of improvements

**Table 2: Operator Performance**
- Columns: Operator, Avg Reward, Selection Frequency
- Shows: Which operators are actually good

---

## Implementation Plan

### Phase 1: Core Infrastructure ✓ IN PROGRESS
**Goal**: Extract selector interface without breaking baseline

1. ✓ Create `OperatorSelector` abstract base class
2. ✓ Extract baseline logic into `AdaptiveWeightSelector`
3. Modify ALNS to accept optional selectors
4. Validate: Baseline behavior preserved

**Time estimate**: 2-3 hours

### Phase 2: MAB Implementation
**Goal**: Add UCB1 and Thompson Sampling

5. Implement `UCB1Selector` class
6. Implement `ThompsonSamplingSelector` class
7. Add statistics tracking (counts, rewards, posteriors)

**Time estimate**: 2-3 hours

### Phase 3: Experiments
**Goal**: Run experiments and generate outputs

8.  Run 3 algorithms × 30 instances (50/100/200/500)
9.  Generate convergence plots and boxplots
10. Generate operator evolution plots
11. Create LaTeX tables with statistics

**Time estimate**: 3-4 hours

**Total**: 7-10 hours

---

## Key Design Decisions Summary

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| Architecture | Strategy vs. Inheritance | **Strategy** | Minimal ALNS changes, better testability |
| Selector scope | Destroy only vs. Both | **Both independently** | Leverages repair diversity, manageable arm count |
| Reward function | Binary vs. Tiered | **Tiered (10/1/0)** | Matches baseline, emphasizes new best |
| MAB algorithms | UCB1 vs. Thompson vs. Others | **Both** | UCB1 is standard; Thompson better for non-stationary |
| Instance sizes | 50/100/200/500 | **50/100/200** | 500 too slow for extensive testing |
| Algorithm count | 5+ vs. 3 | **3 only** | Time-constrained; focus on quality over quantity |

---

## Risk Mitigation

### Risk 1: MAB Doesn't Improve Over Baseline
**Likelihood**: Medium
**Impact**: Report contribution
**Mitigation**: Frame as validation study; discuss why SCFPDP may not benefit (only 3 operators = limited exploration benefit)

### Risk 2: Only 3 Operators Per Type
**Concern**: MAB advantage might be small with few operators
**Mitigation**: Papers with 8+ operators showed clearer benefit; but validates on realistic problem

### Risk 3: Statistical Power
**Concern**: 30 instances might not be enough
**Mitigation**: Use non-parametric tests (Wilcoxon); report effect sizes not just p-values

---

## Success Criteria

**Minimum viable**:
1. Clean abstraction (OperatorSelector interface)
2. Three working algorithms
3. Statistical comparison on 30 instances
4. Two plots + two tables for report

**Stretch goals** (if time):
- Test on 500-size instances
- Sensitivity analysis on UCB exploration parameter
- Implement additional destroy operators for diversity

---

## Files Created/Modified

**Created** (~300 lines total):
- `src/alns/selectors/__init__.py`
- `src/alns/selectors/base.py` - OperatorSelector ABC
- `src/alns/selectors/adaptive_weights.py` - Baseline
- `src/alns/selectors/mab.py` - UCB1 + Thompson Sampling

**Modified** (~30 lines):
- `src/alns/alns.py` - Add selector injection

**Minimal code footprint, maximum flexibility.**