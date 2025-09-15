"""
Train Delay Analysis Module

This module provides functionality to analyze train delay patterns and generate
statistics for use in train scheduling systems.

Usage:
    from delay_analyzer import analyze_train_delays
    
    schedule_data = [
        {"train_id": "train_1", "delay": 2.5},
        {"train_id": "train_2", "delay": 0.0},
        # ... more data
    ]
    
    results = analyze_train_delays(schedule_data)
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_train_delays(schedule_data: List[Dict[str, Any]], 
                        outlier_threshold: float = 2.0,
                        delay_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Analyze train delay patterns and generate statistics.
    
    This function processes historical train delay data to calculate key metrics
    including mean delays, standard deviations, and delay probabilities per train.
    
    Args:
        schedule_data (List[Dict]): List of dictionaries containing train data.
                                  Each dict should have 'train_id' and 'delay' keys.
        outlier_threshold (float): Z-score threshold for outlier removal (default: 2.0)
        delay_threshold (float): Minimum delay in minutes to consider as "delayed" (default: 1.0)
    
    Returns:
        Dict[str, Any]: Dictionary containing delay statistics:
            - mean_delay: Overall mean delay across all trains
            - std_delay: Overall standard deviation of delays
            - total_trains: Total number of train records
            - delayed_trains: Number of trains with delays > threshold
            - delay_probabilities: Dict mapping train_id to probability of delay
            - per_train_stats: Detailed statistics for each train
            - data_quality: Information about data processing
    
    Example:
        >>> data = [{"train_id": "A", "delay": 2.5}, {"train_id": "B", "delay": 0.0}]
        >>> results = analyze_train_delays(data)
        >>> print(results["mean_delay"])
        1.25
    """
    
    # Input validation and graceful error handling
    if not schedule_data:
        logger.warning("Empty schedule_data provided")
        return _empty_response()
    
    if not isinstance(schedule_data, list):
        logger.error("schedule_data must be a list")
        return _empty_response()
    
    # Parse and validate input data
    parsed_data = _parse_input_data(schedule_data)
    if not parsed_data:
        logger.warning("No valid data found after parsing")
        return _empty_response()
    
    # Group delays by train_id
    train_delays = _group_delays_by_train(parsed_data)
    
    # Remove outliers for more robust statistics
    cleaned_delays = _remove_outliers(train_delays, outlier_threshold)
    
    # Calculate overall statistics
    all_delays = []
    for delays in cleaned_delays.values():
        all_delays.extend(delays)
    
    if not all_delays:
        logger.warning("No delays remaining after outlier removal")
        return _empty_response()
    
    # Calculate main metrics
    mean_delay = float(np.mean(all_delays))
    std_delay = float(np.std(all_delays))
    total_trains = len(parsed_data)
    delayed_trains = sum(1 for item in parsed_data if item['delay'] > delay_threshold)
    
    # Calculate per-train statistics and delay probabilities
    per_train_stats = {}
    delay_probabilities = {}
    
    for train_id, delays in train_delays.items():
        cleaned = cleaned_delays.get(train_id, [])
        
        # Calculate statistics for this train
        if cleaned:
            train_mean = float(np.mean(cleaned))
            train_std = float(np.std(cleaned))
            delay_prob = float(np.mean(np.array(cleaned) > delay_threshold))
        else:
            train_mean = 0.0
            train_std = 0.0
            delay_prob = 0.0
        
        per_train_stats[train_id] = {
            'mean_delay': train_mean,
            'std_delay': train_std,
            'total_records': len(delays),
            'records_after_cleaning': len(cleaned),
            'delay_probability': delay_prob
        }
        
        delay_probabilities[train_id] = delay_prob
    
    # Data quality metrics
    original_count = len(parsed_data)
    cleaned_count = len(all_delays)
    outliers_removed = original_count - cleaned_count
    
    return {
        'mean_delay': round(mean_delay, 2),
        'std_delay': round(std_delay, 2),
        'total_trains': total_trains,
        'delayed_trains': delayed_trains,
        'delay_probabilities': {k: round(v, 3) for k, v in delay_probabilities.items()},
        'per_train_stats': {
            k: {
                'mean_delay': round(v['mean_delay'], 2),
                'std_delay': round(v['std_delay'], 2),
                'total_records': v['total_records'],
                'records_after_cleaning': v['records_after_cleaning'],
                'delay_probability': round(v['delay_probability'], 3)
            }
            for k, v in per_train_stats.items()
        },
        'data_quality': {
            'original_records': original_count,
            'records_after_cleaning': cleaned_count,
            'outliers_removed': outliers_removed,
            'outlier_threshold_used': outlier_threshold,
            'delay_threshold_used': delay_threshold
        }
    }


def _parse_input_data(schedule_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse and validate input data, filtering out invalid records.
    
    Args:
        schedule_data: Raw input data
        
    Returns:
        List of valid data records
    """
    valid_data = []
    
    for i, item in enumerate(schedule_data):
        try:
            # Check required fields
            if not isinstance(item, dict):
                logger.warning(f"Record {i}: Not a dictionary, skipping")
                continue
                
            if 'train_id' not in item or 'delay' not in item:
                logger.warning(f"Record {i}: Missing required fields (train_id, delay)")
                continue
            
            # Convert delay to float
            delay = float(item['delay'])
            train_id = str(item['train_id'])
            
            # Basic validation
            if delay < 0:
                logger.warning(f"Record {i}: Negative delay {delay}, setting to 0")
                delay = 0.0
            
            if not train_id.strip():
                logger.warning(f"Record {i}: Empty train_id, skipping")
                continue
            
            valid_data.append({
                'train_id': train_id,
                'delay': delay
            })
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Record {i}: Invalid data format - {e}")
            continue
    
    return valid_data


def _group_delays_by_train(data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Group delays by train_id.
    
    Args:
        data: List of validated data records
        
    Returns:
        Dictionary mapping train_id to list of delays
    """
    train_delays = {}
    
    for item in data:
        train_id = item['train_id']
        delay = item['delay']
        
        if train_id not in train_delays:
            train_delays[train_id] = []
        
        train_delays[train_id].append(delay)
    
    return train_delays


def _remove_outliers(train_delays: Dict[str, List[float]], 
                    z_threshold: float = 2.0) -> Dict[str, List[float]]:
    """
    Remove outliers from delay data using z-score method.
    
    Args:
        train_delays: Dictionary of train_id -> delays
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        Dictionary with outliers removed
    """
    cleaned_delays = {}
    
    for train_id, delays in train_delays.items():
        if len(delays) < 3:
            # Not enough data for outlier detection, keep all
            cleaned_delays[train_id] = delays
        else:
            # Calculate z-scores and filter
            delays_array = np.array(delays)
            z_scores = np.abs(stats.zscore(delays_array))
            clean_mask = z_scores < z_threshold
            cleaned_delays[train_id] = delays_array[clean_mask].tolist()
    
    return cleaned_delays


def _empty_response() -> Dict[str, Any]:
    """
    Return empty response structure for error cases.
    
    Returns:
        Dictionary with empty/zero values
    """
    return {
        'mean_delay': 0.0,
        'std_delay': 0.0,
        'total_trains': 0,
        'delayed_trains': 0,
        'delay_probabilities': {},
        'per_train_stats': {},
        'data_quality': {
            'original_records': 0,
            'records_after_cleaning': 0,
            'outliers_removed': 0,
            'outlier_threshold_used': 2.0,
            'delay_threshold_used': 1.0
        }
    }


# Example usage and testing function
def test_analyzer():
    """Test function to demonstrate usage."""
    test_data = [
        {"train_id": "train_1", "delay": 2.5},
        {"train_id": "train_1", "delay": 3.0},
        {"train_id": "train_2", "delay": 0.0},
        {"train_id": "train_2", "delay": 1.2},
        {"train_id": "train_3", "delay": 5.5},
        {"train_id": "train_1", "delay": 15.0},  # Outlier
    ]
    
    results = analyze_train_delays(test_data)
    print("Test Results:")
    print(f"Mean delay: {results['mean_delay']} minutes")
    print(f"Total trains: {results['total_trains']}")
    print(f"Delayed trains: {results['delayed_trains']}")
    print(f"Delay probabilities: {results['delay_probabilities']}")
    
    return results


if __name__ == "__main__":
    test_analyzer()