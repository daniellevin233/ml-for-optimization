from collections import defaultdict

from src.instance import SocialGolferInstance
from src.sat import SATSocialGolferSolver


class Group:
    def __init__(self, golfers: list[int]):
        self.golfers = golfers

    def __repr__(self):
        return f"Golfers({self.golfers})"


class Week:
    def __init__(self, groups: list[Group]):
        self.groups = groups

    def __repr__(self):
        return f"Week({self.groups})"


class SocialGolferSolution:
    def __init__(self, instance: SocialGolferInstance):
        self.instance = instance
        self.weeks: list[Week] = []

    def __repr__(self):
        if not self.weeks:
            return "SocialGolferSolution(empty)"

        try:
            self.validate()
        except ValueError as e:
            return f"SocialGolferSolution(INVALID): {e}"

        lines = [
            f"Social Golfer Solution: {self.instance.m} groups × {self.instance.n} golfers × {len(self.weeks)} weeks",
            ""
        ]

        # Calculate column widths
        num_weeks = len(self.weeks)
        num_groups = len(self.weeks[0].groups) if self.weeks else 0
        # Width = 2 (braces) + 2*n (digits) + 2*(n-1) (separators) = 4*n
        # Plus 1 for leading space
        n = self.instance.n
        col_width = 4 * n + 1
        group_col_width = 9

        # Header row (weeks as columns)
        header = " " * group_col_width + "│"
        for l in range(1, num_weeks + 1):
            header += f" Week {l}".ljust(col_width) + "│"
        lines.append(header)

        # Separator
        sep = "─" * group_col_width + "┼" + ("─" * col_width + "┼") * num_weeks
        lines.append(sep)

        # Data rows (groups as rows)
        for k in range(num_groups):
            row = f" Group {k + 1}".ljust(group_col_width) + "│"
            for week in self.weeks:
                group = week.groups[k]
                golfers_str = "{" + ", ".join(str(g) for g in group.golfers) + "}"
                row += f" {golfers_str}".ljust(col_width) + "│"
            lines.append(row)

        return "\n".join(lines)

    def copy_from(self, other_solution: 'SocialGolferSolution'):
        self.weeks = other_solution.weeks.copy()

    def construct_with_sat_solver(self):
        sat_solver = SATSocialGolferSolver(instance=self.instance)
        self.copy_from(sat_solver.get_solution())

    def validate(self):
        played_with = defaultdict(set)  # golfer_id -> set of golfer_ids played with
        if len(self.weeks) != self.instance.w:
            raise ValueError(f"Wrong number of weeks in the solution: {len(self.weeks)} != {self.instance.w}")
        for w, week in enumerate(self.weeks, start=1):
            for group in week.groups:
                if len(group.golfers) != self.instance.n:
                    raise ValueError(f"Group {group.golfers} has wrong number of golfers: {len(group.golfers)} != {self.instance.n}")
                for i in range(len(group.golfers)):
                    for j in range(i + 1, len(group.golfers)):
                        g1 = group.golfers[i]
                        g2 = group.golfers[j]
                        if g2 in played_with[g1]:
                            raise ValueError(f"Week {w} assignment is invalid: {g1} has already played with {g2}")
                        played_with[g1].add(g2)
                        played_with[g2].add(g1)

if __name__ == '__main__':
    # solution = SocialGolferSolution(SocialGolferInstance('sgp_1_3_1.txt'))
    # solution.construct_with_sat_solver()
    # print(solution)

    # solution = SocialGolferSolution(SocialGolferInstance('sgp_2_2_3.txt'))
    # noinspection LanguageDetectionInspection
    solution = SocialGolferSolution(SocialGolferInstance('sgp_8_4_7.txt'))
    solution.construct_with_sat_solver()
    # solution.weeks[-1].groups[0].golfers = [4, 3]
    # solution.weeks[-1].groups[1].golfers = [2, 1]
    print(solution)

    # solution = SocialGolferSolution(SocialGolferInstance('sgp_1_3_2.txt'))
    # solution.construct_with_sat_solver()
    # print(solution)