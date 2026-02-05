"""DQN selector for ALNS - simple and interpretable implementation."""

import copy
import random
from collections import deque
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.alns.operators import DestroyOperator, RepairOperator


class QNetwork(nn.Module):
    """Simple 2-layer MLP: 7 inputs → 64 → 64 → 144 outputs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 144)  # 3*3*4*4 = 144 actions
        )

    def forward(self, x):
        return self.net(x)


class DQNSelector:
    """
    DQN selector controlling operators + parameters.

    Returns: (destroy_op, repair_op, severity, temp_multiplier)

    State (7 features): normalized objectives, temperature, stagnation, acceptance/improvement rates
    Actions (144): 3 destroy × 3 repair × 4 severity × 4 temp
    """

    def __init__(self, destroy_ops: list[DestroyOperator], repair_ops: list[RepairOperator]):
        self.destroy_ops = destroy_ops
        self.repair_ops = repair_ops
        self.severity_levels = [0.1, 0.2, 0.3, 0.4]
        self.temp_multipliers = [0.5, 1.0, 1.5, 2.0]

        self.q_network = QNetwork()
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        self.buffer = deque(maxlen=10000)
        self.epsilon = 1.0

        self.current_state = None
        self.last_action = None
        self.steps = 0

    def _extract_state(self, state_dict: Dict) -> np.ndarray:
        curr = state_dict["current_objective"]
        best = state_dict["best_objective"]
        init = state_dict["initial_objective"]
        temp = state_dict["temperature"]
        init_temp = state_dict["initial_temperature"]
        no_improve = state_dict["iterations_without_improvement"]
        max_iter = state_dict["max_iterations"]
        stats = state_dict["statistics"]

        # Acceptance/improvement rates (last 100 iterations)
        recent = stats[-100:] if len(stats) >= 100 else stats
        acc_rate = sum(s.accepted for s in recent) / max(len(recent), 1)
        imp_rate = sum(s.improvement for s in recent) / max(len(recent), 1)

        return np.array([
            curr / max(init, 1),
            best / max(init, 1),
            curr / max(best, 1),
            min(no_improve / max(max_iter, 1), 1.0),
            temp / max(init_temp, 1),
            acc_rate,
            imp_rate
        ], dtype=np.float32)

    def _decode_action(self, idx: int) -> Tuple[int, int, int, int]:
        """Convert flat action index to (destroy, repair, severity, temp) indices."""
        temp_idx = idx // 36  # 36 = 3*3*4
        idx = idx % 36
        severity_idx = idx // 9  # 9 = 3*3
        idx = idx % 9
        repair_idx = idx // 3
        destroy_idx = idx % 3
        return destroy_idx, repair_idx, severity_idx, temp_idx

    def select(self, state: Dict):
        """Epsilon-greedy action selection. Returns (destroy_op, repair_op, severity, temp_mult)."""
        self.current_state = self._extract_state(state)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, 143)
        else:
            with torch.no_grad():
                q_vals = self.q_network(torch.FloatTensor(self.current_state))
                action_idx = q_vals.argmax().item()

        # Decode
        d_idx, r_idx, s_idx, t_idx = self._decode_action(action_idx)
        self.last_action = action_idx

        return (
            self.destroy_ops[d_idx],
            self.repair_ops[r_idx],
            self.severity_levels[s_idx],
            self.temp_multipliers[t_idx]
        )

    def update(self, _operator: Any, reward: float, next_state: Dict):
        """Store transition and train."""
        next_state_vec = self._extract_state(next_state)
        self.buffer.append((self.current_state, self.last_action, reward, next_state_vec))

        if len(self.buffer) >= 64:
            batch = random.sample(self.buffer, 64)
            states, actions, rewards, next_states = zip(*batch)

            s = torch.FloatTensor(np.array(states))
            a = torch.LongTensor(actions)
            r = torch.FloatTensor(rewards)
            s_next = torch.FloatTensor(np.array(next_states))

            # Q-learning update
            q_current = self.q_network(s).gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                q_next = self.target_network(s_next).max(1)[0]
                q_target = r + 0.99 * q_next

            loss = ((q_current - q_target) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(0.1, self.epsilon * 0.995)

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def get_statistics(self) -> Dict:
        """Return stats for tracking."""
        d_idx, r_idx, s_idx, t_idx = self._decode_action(self.last_action) if self.last_action is not None else (0, 0, 0, 0)
        return {
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "destroy_op_chosen": self.destroy_ops[d_idx].name,
            "repair_op_chosen": self.repair_ops[r_idx].name,
            "severity_chosen": self.severity_levels[s_idx],
            "temp_multiplier_chosen": self.temp_multipliers[t_idx],
        }


def train_dqn():
    """
    Train DQN policy on all train instances.

    Each instance = 1 training episode.
    DQN learns sequentially across all episodes.
    """
    import pickle
    from tqdm import tqdm
    from src.alns.alns import ALNS
    from src.alns.config import ALNSConfig
    from src.alns.operators import create_all_destroy_operators, create_all_repair_operators
    from src.construction_heuristics import FlexiblePickupAndDropoffConstructionHeuristic
    from src.instance import SCFPDPInstance
    from src.solution import SCFPDPSolution
    from src.utils import find_project_root

    project_root = find_project_root()

    # Load ALL train instances (50 and 100)
    train_paths = []
    train_paths.extend(sorted((project_root / "scfpdp_instances/50/train").glob("*.txt")))
    train_paths.extend(sorted((project_root / "scfpdp_instances/100/train").glob("*.txt")))

    print(f"Training DQN on {len(train_paths)} instances")
    print(f"  - Size 50: {len([p for p in train_paths if '/50/' in str(p)])} instances")
    print(f"  - Size 100: {len([p for p in train_paths if '/100/' in str(p)])} instances")

    # Create shared DQN selector
    config = ALNSConfig(max_iterations=500, max_time_seconds=100, log_interval=10000)
    destroy_ops = create_all_destroy_operators(config)
    repair_ops = create_all_repair_operators(config)
    dqn = DQNSelector(destroy_ops, repair_ops)

    episode_data = []

    # Train sequentially (DQN learns across episodes)
    for instance_path in tqdm(train_paths, desc="Training DQN"):
        instance = SCFPDPInstance(str(instance_path))

        # Initial solution
        solution = SCFPDPSolution(instance)
        FlexiblePickupAndDropoffConstructionHeuristic(solution).construct()
        initial_obj = solution.calc_objective()

        # Run ALNS with DQN
        alns = ALNS(solution, config=config, destroy_selector=dqn)
        best = alns.run()
        final_obj = best.calc_objective()

        improvement = 100 * (initial_obj - final_obj) / initial_obj
        episode_data.append({
            "instance": instance_path.name,
            "size": instance.n,
            "initial_obj": initial_obj,
            "final_obj": final_obj,
            "improvement": improvement,
            "epsilon": dqn.epsilon,
        })

    # Save trained policy and training data
    results_dir = project_root / "trained_models/dqn"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(dqn.q_network.state_dict(), results_dir / "dqn_policy.pth")
    with open(results_dir / "dqn_training_data.pkl", "wb") as f:
        pickle.dump(episode_data, f)

    print(f"\n✓ Training complete!")
    print(f"  Saved policy: {results_dir / 'dqn_policy.pth'}")
    avg_improvement = sum(e["improvement"] for e in episode_data) / len(episode_data)
    print(f"  Avg improvement: {avg_improvement:.1f}%")
    print(f"  Final epsilon: {dqn.epsilon:.3f}")

    return dqn

if __name__ == '__main__':
    train_dqn()
