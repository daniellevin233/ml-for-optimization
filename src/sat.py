from typing import TYPE_CHECKING, Any

from pysat.formula import CNF
from pysat.solvers import Solver

from src.instance import SocialGolferInstance

if TYPE_CHECKING:
    from src.solution import SocialGolferSolution


class SATSocialGolferSolver:
    def __init__(self, instance: SocialGolferInstance):
        self.instance = instance
        self.cnf = self._build_cnf()

    def __repr__(self):
        try:
            model, time = self.get_model()
            return (f"{self.instance} is Satisfiable.\n"
                    f"Solved in {time:.4f} seconds.")
                    # f"Model: {model}")
        except ValueError:
            return f"{self.instance} is Unsatisfiable."

    # Variable encoding helpers
    # G_ijkl -> unique positive integer
    def G(self, i, j, k, l):
        """G_ijkl: player i in position j of group k in week l (1-indexed)"""
        x = self.instance.total_golfers  # total golfers
        g = self.instance.m              # number of groups
        p = self.instance.n              # golfers per group
        w = self.instance.w              # number of weeks
        return ((l - 1) * g * p * x +
                (k - 1) * p * x +
                (j - 1) * x +
                i)

    def _build_cnf(self) -> CNF:
        """
        Build CNF using the corrected SAT formulation from:
        "An improved SAT formulation for the social golfer problem"
        Triska & Musliu, Ann Oper Res (2012) 194:427-438

        Section 4: Revisiting the SAT formulation by Gent and Lynce

        Paper notation (mapped to instance attributes):
            x = total golfers = g × p     -> self.instance.total_golfers
            g = number of groups          -> self.instance.m
            p = golfers per group         -> self.instance.n
            w = number of weeks           -> self.instance.w

        Variables:
            G_ijkl: player i plays in position j of group k in week l
                1 ≤ i ≤ x, 1 ≤ j ≤ p, 1 ≤ k ≤ g, 1 ≤ l ≤ w

            G'_ikl: player i plays in group k in week l (auxiliary)
                1 ≤ i ≤ x, 1 ≤ k ≤ g, 1 ≤ l ≤ w

            LADDER_yz: ladder matrix for socialization constraint
                1 ≤ y ≤ C(x,2), 1 ≤ z ≤ g×w+1
        """
        x = self.instance.total_golfers  # total golfers
        g = self.instance.m              # number of groups
        p = self.instance.n              # golfers per group
        w = self.instance.w              # number of weeks

        clauses = []

        num_G_vars = x * p * g * w

        # G'_ikl -> unique positive integer (offset by num_G_vars)
        def G_prime(i, k, l):
            """G'_ikl: player i plays in group k in week l"""
            return num_G_vars + ((l - 1) * g * x +
                                  (k - 1) * x +
                                  i)

        num_G_prime_vars = x * g * w

        # LADDER_yz -> unique positive integer (offset by num_G_vars + num_G_prime_vars)
        def pair_index(i, m):
            """Index for pair (i, m) where i < m, using formula C(x,2) - C(x-i,2) + (m-i)"""
            # Row index for pair (i, m): (x - i choose 2) + m - i
            # Using: sum from 1 to i-1 of (x-k) = (i-1)*x - (i-1)*i/2
            return (i - 1) * x - (i - 1) * i // 2 + (m - i)

        num_pairs = x * (x - 1) // 2  # C(x, 2)
        num_ladder_cols = g * w + 1

        def LADDER(y, z):
            """LADDER_yz: ladder matrix variable (1-indexed y and z)"""
            return num_G_vars + num_G_prime_vars + (y - 1) * num_ladder_cols + z

        # ============================================================
        # Clause (1): Each player plays at least once per week
        # ∧(i=1..x) ∧(l=1..w) ∨(j=1..p) ∨(k=1..g) G_ijkl
        # ============================================================
        for i in range(1, x + 1):
            for l in range(1, w + 1):
                clause = []
                for j in range(1, p + 1):
                    for k in range(1, g + 1):
                        clause.append(self.G(i, j, k, l))
                clauses.append(clause)

        # ============================================================
        # Clause (2): Each player plays at most once per group per week
        # ∧(i=1..x) ∧(l=1..w) ∧(j=1..p) ∧(k=1..g) ∧(m=j+1..p) ¬G_ijkl ∨ ¬G_imkl
        # ============================================================
        for i in range(1, x + 1):
            for l in range(1, w + 1):
                for j in range(1, p + 1):
                    for k in range(1, g + 1):
                        for m in range(j + 1, p + 1):
                            clauses.append([-self.G(i, j, k, l), -self.G(i, m, k, l)])

        # ============================================================
        # Clause (12): Each player plays in at most one group per week
        # ∧(i=1..x) ∧(l=1..w) ∧(j=1..p) ∧(k=1..g) ∧(m=k+1..g) ∧(n=1..p) ¬G_ijkl ∨ ¬G_inml
        # ============================================================
        for i in range(1, x + 1):
            for l in range(1, w + 1):
                for j in range(1, p + 1):
                    for k in range(1, g + 1):
                        for m in range(k + 1, g + 1):
                            for n in range(1, p + 1):
                                clauses.append([-self.G(i, j, k, l), -self.G(i, n, m, l)])

        # ============================================================
        # Clause (4): At least one player is the j-th golfer in each group
        # ∧(l=1..w) ∧(k=1..g) ∧(j=1..p) ∨(i=1..x) G_ijkl
        # ============================================================
        for l in range(1, w + 1):
            for k in range(1, g + 1):
                for j in range(1, p + 1):
                    clause = []
                    for i in range(1, x + 1):
                        clause.append(self.G(i, j, k, l))
                    clauses.append(clause)

        # ============================================================
        # Clause (13): At most one player is the j-th golfer
        # ∧(l=1..w) ∧(k=1..g) ∧(j=1..p) ∧(i=1..x) ∧(m=i+1..x) ¬G_ijkl ∨ ¬G_mjkl
        # ============================================================
        for l in range(1, w + 1):
            for k in range(1, g + 1):
                for j in range(1, p + 1):
                    for i in range(1, x + 1):
                        for m in range(i + 1, x + 1):
                            clauses.append([-self.G(i, j, k, l), -self.G(m, j, k, l)])

        # ============================================================
        # Clause (6): Auxiliary variable equivalence
        # G'_ikl ↔ ∨(j=1..p) G_ijkl
        # Encoded as:
        #   (a) G'_ikl → ∨(j=1..p) G_ijkl  (if G'_ikl then some G_ijkl)
        #   (b) ∧(j=1..p) G_ijkl → G'_ikl  (if any G_ijkl then G'_ikl)
        # ============================================================
        for i in range(1, x + 1):
            for k in range(1, g + 1):
                for l in range(1, w + 1):
                    # (a) ¬G'_ikl ∨ G_i1kl ∨ G_i2kl ∨ ... ∨ G_ipkl
                    clause = [-G_prime(i, k, l)]
                    for j in range(1, p + 1):
                        clause.append(self.G(i, j, k, l))
                    clauses.append(clause)

                    # (b) For each j: ¬G_ijkl ∨ G'_ikl
                    for j in range(1, p + 1):
                        clauses.append([-self.G(i, j, k, l), G_prime(i, k, l)])

        # ============================================================
        # Clause (14): Ladder validity constraint
        # ∧(y=1..C(x,2)) ∧(z=1..g×w) ¬LADDER_y(z+1) ∨ LADDER_yz
        # Ensures each row is T...TF...F pattern
        # ============================================================
        for y in range(1, num_pairs + 1):
            for z in range(1, g * w + 1):
                clauses.append([-LADDER(y, z + 1), LADDER(y, z)])

        # ============================================================
        # Clauses (17-20): Socialisation via ladder matrix
        # For pair (i, m) with i < m:
        #   y = pair_index(i, m) = C(x,2) - C(x-i+1,2) + (m-i)
        #   col = (l-1)×g+k
        # ============================================================
        for l in range(1, w + 1):
            for k in range(1, g + 1):
                col = (l - 1) * g + k  # unique column index
                for i in range(1, x):
                    for m in range(i + 1, x + 1):
                        y = pair_index(i, m)

                        # (17): If both play in same group, ladder must be true
                        # ¬G'_ikl ∨ ¬G'_mkl ∨ LADDER_y,col
                        clauses.append([-G_prime(i, k, l), -G_prime(m, k, l), LADDER(y, col)])

                        # (18): If both play in same group, next ladder must be false
                        # ¬G'_ikl ∨ ¬G'_mkl ∨ ¬LADDER_y,col+1
                        clauses.append([-G_prime(i, k, l), -G_prime(m, k, l), -LADDER(y, col + 1)])

                        # (19): If there's a T→F transition at some column, both players must be in that group
                        # LADDER_y,col+1 ∨ ¬LADDER_y,col ∨ G'_ikl
                        clauses.append([LADDER(y, col + 1), -LADDER(y, col), G_prime(i, k, l)])

                        # (20): If there's a T→F transition at some column, both players must be in that group
                        # LADDER_y,col+1 ∨ ¬LADDER_y,col ∨ G'_mkl
                        clauses.append([LADDER(y, col + 1), -LADDER(y, col), G_prime(m, k, l)])

        cnf = CNF(from_clauses=clauses)
        return cnf

    def is_satisfiable(self) -> tuple[bool, float]:
        with Solver(bootstrap_with=self.cnf, use_timer=True) as solver:
            return solver.solve(), solver.time()

    def get_model(self) -> tuple[Any, float]:
        with Solver(bootstrap_with=self.cnf, use_timer=True) as solver:
            if solver.solve():
                return solver.get_model(), solver.time()
            else:
                raise ValueError("No satisfying variables assignments found")

    def get_solution(self) -> "SocialGolferSolution":
        """
        Convert the SAT model to a SocialGolferSolution.

        Decodes G_ijkl variables from the model:
            G(i, j, k, l) = (l-1)*g*p*x + (k-1)*p*x + (j-1)*x + i

        Where:
            i = player (1..x)
            j = position within group (1..p)
            k = group (1..g)
            l = week (1..w)
        """
        from src.solution import SocialGolferSolution, Group, Week

        model, _ = self.get_model()
        model_set = set(model)

        x = self.instance.total_golfers
        g = self.instance.m
        p = self.instance.n
        w = self.instance.w

        solution = SocialGolferSolution(self.instance)

        for l in range(1, w + 1):
            groups = []
            for k in range(1, g + 1):
                golfers = []
                for j in range(1, p + 1):
                    for i in range(1, x + 1):
                        var = self.G(i, j, k, l)
                        if var in model_set:
                            golfers.append(i)
                            break
                groups.append(Group(golfers))
            solution.weeks.append(Week(groups))

        return solution


if __name__ == '__main__':
    instance = SocialGolferInstance('sgp_1_3_1.txt')
    print(SATSocialGolferSolver(instance))

    instance = SocialGolferInstance('sgp_1_3_2.txt')
    print(SATSocialGolferSolver(instance))

    instance = SocialGolferInstance('sgp_2_2_3.txt')
    print(SATSocialGolferSolver(instance))

    instance = SocialGolferInstance('sgp_5_3_6.txt')
    print(SATSocialGolferSolver(instance))
