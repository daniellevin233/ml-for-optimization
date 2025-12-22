import random
import math
from collections import defaultdict
import itertools
from typing import List, Dict

class SocialGolferScheduler:
    def __init__(self, golfers: int = 32, group_size: int = 4, weeks: int = 10):
        """
        Initialize the Social Golfer Problem
        
        Parameters:
        -----------
        golfers: int - Number of golfers (default 32)
        group_size: int - Golfers per group (default 4)
        weeks: int - Weeks to schedule (default 10)
        """
        self.golfers = golfers
        self.group_size = group_size
        self.weeks = weeks
        
        # Calculate number of groups per week
        self.groups_per_week = golfers // group_size
        
        if golfers % group_size != 0:
            raise ValueError(f"Number of golfers ({golfers}) must be divisible by group size ({group_size})")
        
        # Generate all possible pairs for scoring
        self.all_pairs = list(itertools.combinations(range(golfers), 2))
        
        print(f"Social Golfer Problem: {golfers} golfers, {group_size} per group, {weeks} weeks")
        print(f"Groups per week: {self.groups_per_week}")
        print(f"Total pairs to track: {len(self.all_pairs)}")
    
    def generate_initial_solution(self) -> List[List[List[int]]]:
        """
        Generate a random initial schedule
        
        Returns:
        --------
        schedule: 3D list [weeks][groups][golfers]
        """
        schedule = []
        golfers_list = list(range(self.golfers))
        
        for week in range(self.weeks):
            # Shuffle golfers for this week
            random.shuffle(golfers_list)
            
            # Create groups
            week_groups = []
            for i in range(0, self.golfers, self.group_size):
                group = sorted(golfers_list[i:i + self.group_size])
                week_groups.append(group)
            
            schedule.append(week_groups)
        
        return schedule
    
    def evaluate_schedule(self, schedule: List[List[List[int]]]) -> Dict:
        """
        Evaluate the quality of a schedule
        
        Returns:
        --------
        score_dict: Dictionary with various metrics
        """
        # Count how many times each pair appears together
        pair_counts = defaultdict(int)
        
        for week in schedule:
            for group in week:
                # Count all pairs in this group
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair = tuple(sorted((group[i], group[j])))
                        pair_counts[pair] += 1
        
        # Calculate metrics
        repeated_pairs = sum(1 for count in pair_counts.values() if count > 1)
        max_repeats = max(pair_counts.values()) if pair_counts else 0
        
        # Score: we want to minimize repeated pairs
        # Higher penalty for more repetitions
        score = 0
        for pair, count in pair_counts.items():
            if count > 1:
                # Quadratic penalty: worse for 3+ repeats
                score += (count - 1) ** 2
        
        return {
            'score': score,
            'repeated_pairs': repeated_pairs,
            'max_repeats': max_repeats,
            'perfect': repeated_pairs == 0
        }
    
    def generate_neighbor(self, schedule: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        Generate a neighboring schedule by making a small change
        
        Strategy: Swap two golfers within the same week (maintains group structure)
        """
        # Deep copy the schedule
        neighbor = [[group.copy() for group in week] for week in schedule]
        
        # Choose a random week
        week_idx = random.randint(0, self.weeks - 1)
        
        # Choose two different random groups in that week
        group1_idx, group2_idx = random.sample(range(self.groups_per_week), 2)
        
        # Choose random golfers from each group
        golfer1_idx = random.randint(0, self.group_size - 1)
        golfer2_idx = random.randint(0, self.group_size - 1)
        
        # Swap the golfers
        golfer1 = neighbor[week_idx][group1_idx][golfer1_idx]
        golfer2 = neighbor[week_idx][group2_idx][golfer2_idx]
        
        neighbor[week_idx][group1_idx][golfer1_idx] = golfer2
        neighbor[week_idx][group2_idx][golfer2_idx] = golfer1
        
        # Keep groups sorted for consistency
        neighbor[week_idx][group1_idx].sort()
        neighbor[week_idx][group2_idx].sort()
        
        return neighbor
    
    def simulated_annealing(self, max_iterations: int = 10000, 
                           initial_temp: float = 10.0, 
                           cooling_rate: float = 0.995,
                           verbose: bool = True) -> Dict:
        """
        Run simulated annealing to find a good schedule
        
        Returns:
        --------
        results: Dictionary with best schedule and statistics
        """
        # Initialize
        current_schedule = self.generate_initial_solution()
        current_eval = self.evaluate_schedule(current_schedule)
        
        best_schedule = current_schedule
        best_eval = current_eval
        
        temperature = initial_temp
        
        # For tracking progress
        progress = []
        
        if verbose:
            print(f"\nStarting Simulated Annealing:")
            print(f"Initial score: {current_eval['score']}")
            print(f"Initial repeated pairs: {current_eval['repeated_pairs']}")
        
        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor_schedule = self.generate_neighbor(current_schedule)
            neighbor_eval = self.evaluate_schedule(neighbor_schedule)
            
            # Calculate cost difference (we want to minimize score)
            delta_cost = neighbor_eval['score'] - current_eval['score']
            
            # Acceptance criterion
            if delta_cost < 0:
                # Better solution
                current_schedule = neighbor_schedule
                current_eval = neighbor_eval
                
                # Check if it's the best so far
                if neighbor_eval['score'] < best_eval['score']:
                    best_schedule = neighbor_schedule
                    best_eval = neighbor_eval
                    
                    if verbose and neighbor_eval['perfect']:
                        print(f"\n🎯 PERFECT SOLUTION FOUND at iteration {iteration}!")
                        return {
                            'schedule': best_schedule,
                            'evaluation': best_eval,
                            'iterations': iteration + 1,
                            'progress': progress
                        }
            
            elif random.random() < math.exp(-delta_cost / temperature):
                # Accept worse solution with probability
                current_schedule = neighbor_schedule
                current_eval = neighbor_eval
            
            # Cool down
            temperature *= cooling_rate
            
            # Record progress every 100 iterations
            if iteration % 100 == 0:
                progress.append({
                    'iteration': iteration,
                    'temperature': temperature,
                    'best_score': best_eval['score'],
                    'current_score': current_eval['score'],
                    'repeated_pairs': best_eval['repeated_pairs']
                })
            
            # Early stopping if we found a perfect solution
            if best_eval['perfect']:
                break
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best score: {best_eval['score']}")
            print(f"Repeated pairs: {best_eval['repeated_pairs']}")
            print(f"Max repeats for any pair: {best_eval['max_repeats']}")
            print(f"Perfect schedule: {'YES' if best_eval['perfect'] else 'NO'}")
        
        return {
            'schedule': best_schedule,
            'evaluation': best_eval,
            'iterations': max_iterations,
            'progress': progress
        }
    
    def print_schedule(self, schedule: List[List[List[int]]], 
                       show_golfers: bool = True,
                       show_evaluation: bool = True):
        """
        Pretty print the schedule
        """
        if show_evaluation:
            eval_result = self.evaluate_schedule(schedule)
            print(f"\n{'='*60}")
            print(f"SCHEDULE EVALUATION")
            print(f"{'='*60}")
            print(f"Score: {eval_result['score']}")
            print(f"Repeated pairs: {eval_result['repeated_pairs']}")
            print(f"Maximum repeats for any pair: {eval_result['max_repeats']}")
            print(f"Perfect schedule (no repeats): {eval_result['perfect']}")
            print(f"{'='*60}")
        
        print(f"\nGOLF SCHEDULE ({self.golfers} golfers, {self.weeks} weeks)")
        print(f"{'='*60}")
        
        for week_idx, week in enumerate(schedule):
            print(f"\nWeek {week_idx + 1}:")
            print(f"{'-'*40}")
            for group_idx, group in enumerate(week):
                if show_golfers:
                    # Format golfers as G01, G02, etc.
                    golfers_str = ', '.join([f"G{g+1:02d}" for g in group])
                    print(f"  Group {group_idx + 1}: {golfers_str}")
                else:
                    print(f"  Group {group_idx + 1}: {group}")
        
        print(f"\n{'='*60}")
    
    def analyze_pair_frequency(self, schedule: List[List[List[int]]]):
        """
        Analyze how many times each pair of golfers plays together
        """
        pair_counts = defaultdict(int)
        
        for week in schedule:
            for group in week:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair = tuple(sorted((group[i], group[j])))
                        pair_counts[pair] += 1
        
        # Print statistics
        print(f"\nPAIR FREQUENCY ANALYSIS")
        print(f"{'='*60}")
        
        counts_distribution = defaultdict(int)
        for count in pair_counts.values():
            counts_distribution[count] += 1
        
        print(f"\nFrequency distribution:")
        for count in sorted(counts_distribution.keys()):
            pairs = counts_distribution[count]
            print(f"  {count} time(s): {pairs} pairs")
        
        # Show problematic pairs (those that meet more than once)
        problematic_pairs = [(pair, count) for pair, count in pair_counts.items() if count > 1]
        if problematic_pairs:
            print(f"\nProblematic pairs (meet more than once):")
            for pair, count in problematic_pairs:
                print(f"  G{pair[0]+1:02d} & G{pair[1]+1:02d}: {count} times")


def run_optimization(golfers=32, group_size=4, weeks=10, runs=5):
    """
    Run the optimization multiple times and return the best result
    """
    best_result = None
    best_score = float('inf')
    
    print(f"Running {runs} optimization runs...")
    print(f"{'='*60}")
    
    for run in range(runs):
        print(f"\nRun {run + 1}/{runs}")
        
        # Create scheduler
        scheduler = SocialGolferScheduler(golfers, group_size, weeks)
        
        # Run simulated annealing with adjusted parameters
        result = scheduler.simulated_annealing(
            max_iterations=5000,  # Fewer iterations per run since we're doing multiple runs
            initial_temp=15.0,
            cooling_rate=0.99,
            verbose=False
        )
        
        score = result['evaluation']['score']
        repeated_pairs = result['evaluation']['repeated_pairs']
        
        print(f"  Score: {score}, Repeated pairs: {repeated_pairs}")
        
        if score < best_score:
            best_score = score
            best_result = result
            scheduler_best = scheduler
            
            # If we found a perfect solution, we can stop early
            if result['evaluation']['perfect']:
                print(f"  ✓ Perfect solution found!")
                break
    
    return scheduler_best, best_result


# Example usage and demonstration
if __name__ == "__main__":
    print("SOCIAL GOLFER PROBLEM OPTIMIZATION")
    print("="*60)
    
    # Run optimization
    scheduler, result = run_optimization(
        golfers=32,
        group_size=4,
        weeks=10,
        runs=3
    )
    
    # Display the best schedule
    print(f"\n{'='*60}")
    print("BEST SCHEDULE FOUND")
    print(f"{'='*60}")
    
    scheduler.print_schedule(result['schedule'])
    scheduler.analyze_pair_frequency(result['schedule'])
    
    # Additional analysis
    print(f"\n{'='*60}")
    print("ADDITIONAL STATISTICS")
    print(f"{'='*60}")
    
    total_golfers = scheduler.golfers
    total_pairs = total_golfers * (total_golfers - 1) // 2
    weeks = scheduler.weeks
    group_size = scheduler.group_size
    
    # Theoretical minimum
    pairs_per_week = (total_golfers // group_size) * (group_size * (group_size - 1) // 2)
    total_meetings_in_schedule = pairs_per_week * weeks
    
    print(f"Total golfers: {total_golfers}")
    print(f"Total possible pairs: {total_pairs}")
    print(f"Pairs meeting per week: {pairs_per_week}")
    print(f"Total meetings in {weeks}-week schedule: {total_meetings_in_schedule}")
    print(f"Average meetings per pair: {total_meetings_in_schedule / total_pairs:.2f}")
    
    # Check if perfect schedule is theoretically possible
    if total_meetings_in_schedule <= total_pairs:
        print(f"\n✓ Perfect schedule (no repeated pairs) is theoretically possible!")
        print(f"  We need to arrange {total_meetings_in_schedule} meetings among {total_pairs} unique pairs.")
    else:
        print(f"\n⚠ Perfect schedule may not be possible")
        print(f"  We need {total_meetings_in_schedule} meetings but only have {total_pairs} unique pairs.")
        print(f"  Minimum repeats: {total_meetings_in_schedule - total_pairs}")
    
    # Example of modifying parameters for different scenarios
    print(f"\n{'='*60}")
    print("TRYING DIFFERENT SCENARIOS")
    print(f"{'='*60}")
    
    # Try a smaller problem that's easier to solve perfectly
    print("\nSmaller problem (12 golfers, 4 per group, 4 weeks):")
    small_scheduler = SocialGolferScheduler(golfers=12, group_size=4, weeks=4)
    small_result = small_scheduler.simulated_annealing(
        max_iterations=3000,
        initial_temp=10.0,
        cooling_rate=0.995,
        verbose=False
    )
    print(f"  Score: {small_result['evaluation']['score']}")
    print(f"  Repeated pairs: {small_result['evaluation']['repeated_pairs']}")
    print(f"  Perfect: {small_result['evaluation']['perfect']}")
    
    # For "full socialization" variant (each pair meets at least once)
    print("\n'Full Socialization' variant (8 golfers, 4 per group, minimum weeks):")
    # We need enough weeks for each pair to meet at least once
    full_scheduler = SocialGolferScheduler(golfers=8, group_size=4, weeks=3)
    full_result = full_scheduler.simulated_annealing(
        max_iterations=2000,
        initial_temp=8.0,
        cooling_rate=0.99,
        verbose=False
    )
    
    # Check if all pairs meet at least once
    pair_counts = defaultdict(int)
    for week in full_result['schedule']:
        for group in week:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair = tuple(sorted((group[i], group[j])))
                    pair_counts[pair] += 1
    
    all_pairs_met = all(count >= 1 for count in pair_counts.values())
    print(f"  All pairs met at least once: {all_pairs_met}")
    print(f"  Weeks used: {full_scheduler.weeks}")