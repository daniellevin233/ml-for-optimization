import numpy as np
import pandas as pd
import random
import math
from typing import Dict, List
import time

class SocialGolferSolver:
    """Simulated Annealing Solver from Task 1"""
    
    def __init__(self, max_iterations=5000, initial_temp=10.0, cooling_rate=0.995):
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def solve(self, golfers, group_size, weeks):
        """Solve using Simulated Annealing"""
        groups_per_week = golfers // group_size
        
        # Generate initial solution
        schedule = []
        all_golfers = list(range(golfers))
        
        for _ in range(weeks):
            random.shuffle(all_golfers)
            week_groups = []
            for i in range(0, golfers, group_size):
                group = sorted(all_golfers[i:i + group_size])
                week_groups.append(group)
            schedule.append(week_groups)
        
        best_schedule = schedule
        best_eval = self._evaluate_schedule(schedule)
        current_schedule = schedule
        current_eval = best_eval
        
        temperature = self.initial_temp
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = self._generate_neighbor(current_schedule, golfers, group_size, groups_per_week)
            neighbor_eval = self._evaluate_schedule(neighbor)
            
            # Acceptance criterion
            delta = neighbor_eval['score'] - current_eval['score']
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_schedule = neighbor
                current_eval = neighbor_eval
                
                if neighbor_eval['score'] < best_eval['score']:
                    best_schedule = neighbor
                    best_eval = neighbor_eval
            
            temperature *= self.cooling_rate
        
        return best_eval
    
    def _generate_neighbor(self, schedule, golfers, group_size, groups_per_week):
        """Generate neighboring schedule"""
        neighbor = [[group.copy() for group in week] for week in schedule]
        week_idx = random.randint(0, len(schedule) - 1)
        
        group1_idx, group2_idx = random.sample(range(groups_per_week), 2)
        golfer1_idx = random.randint(0, group_size - 1)
        golfer2_idx = random.randint(0, group_size - 1)
        
        golfer1 = neighbor[week_idx][group1_idx][golfer1_idx]
        golfer2 = neighbor[week_idx][group2_idx][golfer2_idx]
        
        neighbor[week_idx][group1_idx][golfer1_idx] = golfer2
        neighbor[week_idx][group2_idx][golfer2_idx] = golfer1
        
        neighbor[week_idx][group1_idx].sort()
        neighbor[week_idx][group2_idx].sort()
        
        return neighbor
    
    def _evaluate_schedule(self, schedule):
        """Evaluate schedule quality"""
        pair_counts = {}
        
        for week in schedule:
            for group in week:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair = tuple(sorted((group[i], group[j])))
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        repeated_pairs = sum(1 for count in pair_counts.values() if count > 1)
        max_repeats = max(pair_counts.values()) if pair_counts else 0
        score = sum((count - 1) ** 2 for count in pair_counts.values() if count > 1)
        
        return {
            'score': score,
            'repeated_pairs': repeated_pairs,
            'max_repeats': max_repeats,
            'perfect': repeated_pairs == 0
        }

class DatasetGenerator:
    """Generate complete dataset for Social Golfer Problem"""
    
    def __init__(self):
        self.solver = SocialGolferSolver(max_iterations=3000)
        self.dataset = []
    
    def calculate_features(self, golfers: int, group_size: int, weeks: int) -> Dict:
        """Calculate 10 features for a problem instance"""
        groups_per_week = golfers // group_size
        
        # Direct features
        features = {
            'num_golfers': golfers,
            'group_size_feature': group_size,
            'num_weeks': weeks,
            'groups_per_week': groups_per_week
        }
        
        # Derived features
        total_pairs = golfers * (golfers - 1) // 2
        meetings_per_week = groups_per_week * (group_size * (group_size - 1) // 2)
        total_meetings = meetings_per_week * weeks
        pair_coverage_ratio = total_meetings / total_pairs if total_pairs > 0 else 0
        perfect_possible = 1 if total_meetings <= total_pairs else 0
        
        features.update({
            'total_pairs': total_pairs,
            'meetings_per_week': meetings_per_week,
            'total_meetings': total_meetings,
            'pair_coverage_ratio': pair_coverage_ratio,
            'perfect_possible': perfect_possible
        })
        
        # Complexity feature
        search_space_complexity = weeks * (
            math.log(math.factorial(golfers)) - 
            groups_per_week * math.log(math.factorial(group_size))
        )
        features['search_space_complexity'] = search_space_complexity
        
        return features
    
    def generate_instances(self, num_instances: int = 100) -> List[Dict]:
        """Generate random problem instances"""
        instances = []
        
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        for instance_id in range(1, num_instances + 1):
            # Generate valid parameters with controlled distribution
            if instance_id <= 30:
                # First 30: Standard 4-person groups
                group_size = 4
                golfers_options = [8, 12, 16, 20, 24, 28, 32]
                golfers = random.choice(golfers_options)
                while golfers % group_size != 0:
                    golfers = random.choice(golfers_options)
                weeks = random.randint(4, 12)
                
            elif instance_id <= 60:
                # Next 30: 3-person groups
                group_size = 3
                golfers_options = [6, 9, 12, 15, 18, 21, 24, 27, 30]
                golfers = random.choice(golfers_options)
                while golfers % group_size != 0:
                    golfers = random.choice(golfers_options)
                weeks = random.randint(4, 13)
                
            elif instance_id <= 90:
                # Next 30: 5 and 6 person groups
                group_size = random.choice([5, 6])
                if group_size == 5:
                    golfers_options = [10, 15, 20, 25, 30]
                else:
                    golfers_options = [6, 12, 18, 24, 30]
                golfers = random.choice(golfers_options)
                while golfers % group_size != 0:
                    golfers = random.choice(golfers_options)
                weeks = random.randint(4, 10)
                
            else:
                # Last 10: 8-person groups
                group_size = 8
                golfers_options = [16, 24, 32]
                golfers = random.choice(golfers_options)
                while golfers % group_size != 0:
                    golfers = random.choice(golfers_options)
                weeks = random.randint(4, 8)
            
            instances.append({
                'instance_id': instance_id,
                'golfers': golfers,
                'group_size': group_size,
                'weeks': weeks
            })
        
        return instances
    
    def run_simulated_annealing(self, instances: List[Dict]) -> List[Dict]:
        """Run SA on all instances and collect results"""
        results = []
        
        print("Running Simulated Annealing on 100 instances...")
        print("="*60)
        
        for instance in instances:
            instance_id = instance['instance_id']
            golfers = instance['golfers']
            group_size = instance['group_size']
            weeks = instance['weeks']
            
            # Calculate features
            features = self.calculate_features(golfers, group_size, weeks)
            
            # Run SA
            start_time = time.time()
            sa_result = self.solver.solve(golfers, group_size, weeks)
            runtime = time.time() - start_time
            
            # Combine all data
            result = {
                'instance_id': instance_id,
                'golfers': golfers,
                'group_size': group_size,
                'weeks': weeks,
                **features,
                'sa_score': sa_result['score'],
                'sa_repeated_pairs': sa_result['repeated_pairs'],
                'sa_max_repeats': sa_result['max_repeats'],
                'sa_perfect': 1 if sa_result['perfect'] else 0,
                'sa_runtime_seconds': round(runtime, 2)
            }
            
            results.append(result)
            
            if instance_id % 10 == 0:
                print(f"Completed {instance_id}/100 instances")
        
        return results
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset"""
        # Generate instances
        instances = self.generate_instances(100)
        
        # Run SA on all instances
        results = self.run_simulated_annealing(instances)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns for clarity
        columns_order = [
            'instance_id', 'golfers', 'group_size', 'weeks',
            'num_golfers', 'group_size_feature', 'num_weeks', 'groups_per_week',
            'total_pairs', 'meetings_per_week', 'total_meetings',
            'pair_coverage_ratio', 'perfect_possible', 'search_space_complexity',
            'sa_score', 'sa_repeated_pairs', 'sa_max_repeats', 'sa_perfect',
            'sa_runtime_seconds'
        ]
        
        df = df[columns_order]
        
        return df
    
    def analyze_dataset(self, df: pd.DataFrame):
        """Analyze and print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        # Basic statistics
        print(f"\nTotal instances: {len(df)}")
        print(f"Features: {len([c for c in df.columns if 'sa_' not in c]) - 4}")
        
        # Instance parameter distribution
        print("\nInstance Parameter Distribution:")
        print(f"  Golfers: {df['golfers'].min()} to {df['golfers'].max()}")
        print(f"  Group size: {df['group_size'].unique()}")
        print(f"  Weeks: {df['weeks'].min()} to {df['weeks'].max()}")
        
        # SA performance statistics
        print("\nSA Performance Statistics:")
        print(f"  Average score: {df['sa_score'].mean():.2f}")
        print(f"  Average repeated pairs: {df['sa_repeated_pairs'].mean():.2f}")
        print(f"  Perfect schedules: {df['sa_perfect'].sum()} out of {len(df)}")
        print(f"  Average runtime: {df['sa_runtime_seconds'].mean():.2f} seconds")
        
        # Feature statistics
        print("\nFeature Statistics:")
        numeric_features = [
            'total_pairs', 'meetings_per_week', 'total_meetings',
            'pair_coverage_ratio', 'search_space_complexity'
        ]
        
        for feature in numeric_features:
            print(f"  {feature}:")
            print(f"    Min: {df[feature].min():.2f}")
            print(f"    Max: {df[feature].max():.2f}")
            print(f"    Mean: {df[feature].mean():.2f}")
        
        # Perfect possibility analysis
        perfect_possible = df['perfect_possible'].sum()
        actual_perfect = df['sa_perfect'].sum()
        
        print(f"\nPerfect Schedule Analysis:")
        print(f"  Theoretically possible: {perfect_possible} instances")
        print(f"  Actually achieved: {actual_perfect} instances")
        print(f"  Success rate: {actual_perfect/perfect_possible*100:.1f}%")
        
        # Correlation analysis
        print("\nTop Correlations with SA Score:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_score = df[numeric_cols].corr()['sa_score'].abs().sort_values(ascending=False)
        
        for i, (feature, corr) in enumerate(corr_with_score.items()[:6]):
            if feature != 'sa_score':
                print(f"  {feature}: {corr:.3f}")
        
        return df

def main():
    """Generate and save the complete dataset"""
    # Initialize generator
    generator = DatasetGenerator()
    
    # Generate dataset
    print("Generating Social Golfer Problem Dataset...")
    print("="*60)
    
    df = generator.generate_dataset()
    
    # Analyze dataset
    generator.analyze_dataset(df)
    
    # Save to CSV
    output_file = "social_golfer_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nDataset saved to '{output_file}'")
    
    # Also save to JSON for easy reading
    json_file = "social_golfer_dataset.json"
    df.to_json(json_file, orient='records', indent=2)
    print(f"Dataset also saved to '{json_file}'")
    
    return df

# Generate sample data for the first 15 instances
def generate_sample_output():
    """Generate sample output showing first 15 instances"""
    generator = DatasetGenerator()
    instances = generator.generate_instances(15)
    results = generator.run_simulated_annealing(instances[:15])
    
    print("\nSample Dataset (First 15 Instances):")
    print("="*80)
    print(f"{'ID':<3} {'Golfers':<7} {'Group':<5} {'Weeks':<6} {'Score':<6} {'Repeats':<8} {'Perfect':<8} {'Runtime':<8}")
    print("-"*80)
    
    for result in results[:15]:
        print(f"{result['instance_id']:<3} {result['golfers']:<7} {result['group_size']:<5} "
              f"{result['weeks']:<6} {result['sa_score']:<6} {result['sa_repeated_pairs']:<8} "
              f"{'Yes' if result['sa_perfect'] else 'No':<8} {result['sa_runtime_seconds']:<8.2f}")
    
    return results[:15]

if __name__ == "__main__":
    # Uncomment to generate full dataset
    # df = main()
    
    # For demonstration, show sample output
    sample_data = generate_sample_output()
    
    # Print feature names
    print("\n\nFeature Names:")
    print("-"*40)
    features = [
        'num_golfers', 'group_size_feature', 'num_weeks', 'groups_per_week',
        'total_pairs', 'meetings_per_week', 'total_meetings',
        'pair_coverage_ratio', 'perfect_possible', 'search_space_complexity'
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i:2}. {feature}")