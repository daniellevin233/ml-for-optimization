# ML for Optimization - Project 2 Context

## Course Assignment Requirements
- **Course**: Machine Learning for Optimization, WS 2025, TU Wien
- **Deadline**: 15.02.2026, discussion 16-17.02 or 23.02
- **Grading**: Discussion counts as Exam 2 (40% of total points)

### Task
Implement two adaptive large neighborhood methods for SCF-PDP:
1. **LNS + Multi-Armed Bandit**: Select destroy operators based on MAB
2. **LNS + Reinforcement Learning**: Select destroy operators based on RL

Requirements:
- At least 3 destroy operators
- Exact solver as repair operator + additional repair operators
- Compare to the implemented ALNS with adaptive weight mechanism

## Problem: SCF-PDP (Selective Capacitated Fair Pickup and Delivery Problem)
- **Goal**: Design fair, feasible routes for subset of n customer requests
- **Graph**: Complete directed graph G=(V,A) with depot + pickup/dropoff locations
- **Distances**: Euclidean distance rounded up
- **Requests**: n customer requests CR={1,...,n}, each with pickup v↑_i and dropoff v↓_i
- **Fleet**: K vehicles with capacity C, starting from depot
- **Constraint**: Serve at least γ requests (γ ≤ n)
- **Objective**: Minimize Σd(R_k) + ρ·(1-J(R)) where:
  - d(R_k) = total route distance for vehicle k
  - J(R) = Jain fairness index = (Σd(R_k))² / (n_K · Σd(R_k)²)
  - ρ = fairness weight parameter

## Existing ALNS Implementation (from HOT course assignment)
Located in `src/alns/`:

### Destroy Operators (3)
1. **RandomRemoval**: Remove q randomly selected requests
2. **WorstCostRemoval**: Remove requests with highest distance contribution
3. **LongestRouteRemoval**: Remove from longest routes (targets fairness)

### Repair Operators (3)
1. **GreedyRepair**: FlexiblePickupAndDropoff heuristic
2. **RandomGreedyRepair**: Insert at random feasible positions
3. **ObjectiveAwareRepair**: Minimize full objective (distance + fairness)

### Adaptive Weight Mechanism
- Roulette wheel selection based on weights
- Score: 10 for new best, 1 for accepted, 0 for rejected
- Update formula: ρ_i ← ρ_i·(1-γ) + γ·(s_i/a_i)
- Simulated Annealing acceptance criterion

### Key Parameters (tuned with Optuna)
- weight_update_period, reaction_factor
- min/max_removal_percentage
- initial_temperature, cooling_rate
- score_new_best, score_accepted

## Report Structure (report/report.tex)
Concise report with 3-8 plots, informal language. Sections:
1. Problem Description (SCF-PDP) - ~1 page
2. Baseline ALNS Description - ~1 page
3. MAB Operator Selection
4. RL Operator Selection
5. Results & Comparison
6. Conclusion

## Key Files
- `src/alns/alns.py` - Main ALNS loop
- `src/alns/operators.py` - Destroy/Repair operators
- `src/alns/config.py` - Configuration with Optuna tuning support
- `assignment2_report.tex` - Detailed HOT course report (reference)
- `report/report.tex` - ML4Opt report to edit