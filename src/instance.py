from pathlib import Path

from src.utils import find_project_root


class SocialGolferInstance:
    def __init__(self, file_name: str):
        self.m = None  # number of groups
        self.n = None  # number of players per group
        self.w = None  # number of weeks

        self._parse_file(file_name)

    def _parse_file(self, file_name: str):
        """
        File format:
            m n w
        """
        file_path = Path(file_name)
        if not file_path.is_absolute():
            project_root = find_project_root()
            candidate = project_root / "instances" / file_name
            if candidate.exists():
                file_path = candidate
            else:
                file_path = Path("instances") / file_name

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        params = lines[0].split()
        self.m = int(params[0])
        self.n = int(params[1])
        self.w = int(params[2])

    def __repr__(self):
        return f"SocialGolferInstance(n_groups={self.m}, n_players_per_group={self.n}, n_weeks={self.w})"

if __name__ == '__main__':
    instance = SocialGolferInstance('sgp_4_3_3.txt')
    print(instance)