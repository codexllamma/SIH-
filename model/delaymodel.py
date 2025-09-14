import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import os
def analyze_train_delays(schedulet, plot_results=True):
    """
    Analyze historical train delays and generate Monte Carlo scenarios.
    
    Parameters:
    schedulet (pd.DataFrame): DataFrame with columns ['train_id', 'delay']
    plot_results (bool): Whether to display plots
    
    Returns:
    dict: Dictionary containing delay statistics and scenarios for each train
    """
    csv_path = os.path.join(os.path.dirname(__file__), "trainsmall.csv")
    schedulet=pd.read_csv(csv_path)
    schedulet
    # Separate delays by train
    a_delays = []
    b_delays = []
    c_delays = []
    
    for _, row in schedulet.iterrows():
        if row['train_id'] == 'a':
            a_delays.append(row['delay'])
        elif row['train_id'] == 'b':
            b_delays.append(row['delay'])
        elif row['train_id'] == 'c':
            c_delays.append(row['delay'])
    
    train_a_delays = np.array(a_delays)
    train_b_delays = np.array(b_delays)
    train_c_delays = np.array(c_delays)
    
    def remove_outliers(data, z_threshold=2.0):
        if len(data) == 0:
            return data
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores < z_threshold]
    
    # Clean data
    a_clean = remove_outliers(train_a_delays)
    b_clean = remove_outliers(train_b_delays)
    c_clean = remove_outliers(train_c_delays)
    
    # Calculate statistics
    mean_a = np.mean(a_clean) if len(a_clean) >= 3 else np.mean(a_delays) if len(a_delays) > 0 else 0
    std_a = np.std(a_clean) if len(a_clean) >= 3 else np.std(a_delays) if len(a_delays) > 0 else 1
    mean_b = np.mean(b_clean) if len(b_clean) >= 3 else np.mean(b_delays) if len(b_delays) > 0 else 0
    std_b = np.std(b_clean) if len(b_clean) >= 3 else np.std(b_delays) if len(b_delays) > 0 else 1
    mean_c = np.mean(c_clean) if len(c_clean) >= 3 else np.mean(c_delays) if len(c_delays) > 0 else 0
    std_c = np.std(c_clean) if len(c_clean) >= 3 else np.std(c_delays) if len(c_delays) > 0 else 1
    
    # Generate Monte Carlo scenarios
    np.random.seed(42)
    scenarios_a = np.maximum(0, np.random.normal(mean_a, std_a, 1000))
    scenarios_b = np.maximum(0, np.random.normal(mean_b, std_b, 1000))
    scenarios_c = np.maximum(0, np.random.normal(mean_c, std_c, 1000))
    
    if plot_results:
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        x = np.linspace(0, 12, 200)
        
        # Distribution plots
        ax1.plot(x, stats.norm.pdf(x, mean_a, std_a), 'r-', 
                 label=f'Train A: N({mean_a:.1f}, {std_a:.1f})', linewidth=3)
        ax1.plot(x, stats.norm.pdf(x, mean_b, std_b), 'b-', 
                 label=f'Train B: N({mean_b:.1f}, {std_b:.1f})', linewidth=3)
        ax1.plot(x, stats.norm.pdf(x, mean_c, std_c), 'c-', 
                 label=f'Train C: N({mean_c:.1f}, {std_c:.1f})', linewidth=3)
        
        # Historical data points
        if len(train_a_delays) > 0:
            ax1.scatter(train_a_delays, [0.02]*len(train_a_delays), 
                       color='red', s=120, alpha=0.8, marker='^', 
                       label='Train A Historical', zorder=5)
        if len(train_b_delays) > 0:
            ax1.scatter(train_b_delays, [0.01]*len(train_b_delays), 
                       color='blue', s=120, alpha=0.8, marker='v', 
                       label='Train B Historical', zorder=5)
        if len(train_c_delays) > 0:
            ax1.scatter(train_c_delays, [0.015]*len(train_c_delays), 
                       color='green', s=120, alpha=0.8, marker='s', 
                       label='Train C Historical', zorder=5)
        
        ax1.set_xlabel('Delay (minutes)', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title('Normal Distribution Models', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 12)
        
        # Monte Carlo histograms
        ax2.hist(scenarios_a, bins=50, alpha=0.7, label='Train A Monte Carlo', 
                 density=True, color='red', edgecolor='darkred', linewidth=0.5)
        ax2.hist(scenarios_b, bins=50, alpha=0.7, label='Train B Monte Carlo', 
                 density=True, color='blue', edgecolor='darkblue', linewidth=0.5)
        ax2.hist(scenarios_c, bins=50, alpha=0.7, label='Train C Monte Carlo', 
                 density=True, color='green', edgecolor='darkgreen', linewidth=0.5)
        
        ax2.set_xlabel('Delay (minutes)', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('Monte Carlo Results (1000 scenarios)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 12)
        
        plt.tight_layout()
        plt.show()
        
        # Print probabilities
        print("MONTE CARLO PROBABILITIES:")
        thresholds = [3, 5, 7, 10]
        for threshold in thresholds:
            prob_a = np.mean(scenarios_a > threshold)
            prob_b = np.mean(scenarios_b > threshold)
            prob_c = np.mean(scenarios_c > threshold)
            
            print(f"P(delay > {threshold} min): Train A = {prob_a:.1%}, Train B = {prob_b:.1%}, Train C = {prob_c:.1%}")
        
        print("\nMEAN FOR THEIR DELAYS:")
        print("Train A:", f"{mean_a:.2f}", "+-", f"{std_a:.2f}")
        print("Train B:", f"{mean_b:.2f}", "+-", f"{std_b:.2f}")
        print("Train C:", f"{mean_c:.2f}", "+-", f"{std_c:.2f}")
    
    return {
        'train_a': {
            'mean': mean_a,
            'std': std_a,
            'scenarios': scenarios_a,
            'historical': train_a_delays
        },
        'train_b': {
            'mean': mean_b,
            'std': std_b,
            'scenarios': scenarios_b,
            'historical': train_b_delays
        },
        'train_c': {
            'mean': mean_c,
            'std': std_c,
            'scenarios': scenarios_c,
            'historical': train_c_delays
        }
    }


def check_section_conflicts(train_schedule, delay_stats, time_buffer_minutes=5):
    """
    Check for conflicts when trains arrive at sections with delays.
    
    Parameters:
    train_schedule (pd.DataFrame): DataFrame with columns ['train_id', 'section', 'dep_time', 'arrival_time']
    delay_stats (dict): Output from analyze_train_delays function
    time_buffer_minutes (int): Minimum time buffer between trains in same section
    
    Returns:
    dict: Conflict analysis results with flags for RL model intervention
    """
    conflicts = []
    rl_intervention_needed = False
    
    # Convert time strings to datetime if needed
    def parse_time(time_str):
        if isinstance(time_str, str):
            return datetime.strptime(time_str, '%H:%M')
        return time_str
    
    # Create a copy of schedule to work with
    schedule_copy = train_schedule.copy()
    
    # Add predicted delays to departure and arrival times
    for idx, row in schedule_copy.iterrows():
        train_id = row['train_id']
        
        # Get mean delay for this train
        if train_id in delay_stats:
            mean_delay = delay_stats[train_id]['mean']
        else:
            mean_delay = 0
            
        # Calculate new times with delay
        original_dep = parse_time(row['dep_time'])
        original_arr = parse_time(row['arrival_time'])
        
        delayed_dep = original_dep + timedelta(minutes=mean_delay)
        delayed_arr = original_arr + timedelta(minutes=mean_delay)
        
        schedule_copy.at[idx, 'delayed_dep_time'] = delayed_dep
        schedule_copy.at[idx, 'delayed_arr_time'] = delayed_arr
    
    # Check for conflicts section by section
    sections = schedule_copy['section'].unique()
    
    for section in sections:
        section_trains = schedule_copy[schedule_copy['section'] == section].copy()
        section_trains = section_trains.sort_values('delayed_dep_time')
        
        for i in range(len(section_trains)):
            for j in range(i+1, len(section_trains)):
                train1 = section_trains.iloc[i]
                train2 = section_trains.iloc[j]
                
                # Check if train2 departs before train1 arrives (with buffer)
                time_diff = (train2['delayed_dep_time'] - train1['delayed_arr_time']).total_seconds() / 60
                
                if time_diff < time_buffer_minutes:
                    conflict_info = {
                        'section': section,
                        'train1': train1['train_id'],
                        'train2': train2['train_id'],
                        'train1_arrival': train1['delayed_arr_time'],
                        'train2_departure': train2['delayed_dep_time'],
                        'time_gap_minutes': time_diff,
                        'severity': 'HIGH' if time_diff < 0 else 'MEDIUM'
                    }
                    conflicts.append(conflict_info)
                    rl_intervention_needed = True
    
    return {
        'conflicts_detected': len(conflicts) > 0,
        'num_conflicts': len(conflicts),
        'conflict_details': conflicts,
        'rl_intervention_flag': rl_intervention_needed,
        'updated_schedule': schedule_copy
    }


