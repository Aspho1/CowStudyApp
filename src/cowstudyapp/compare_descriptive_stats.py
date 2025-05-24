import json
from statistics import mean
from typing import Any, Dict, List
import numpy as np
import pandas as pd




def compute_gap_stats(gap_distribution: Dict[str, str]) -> Dict[str, float]:
    """
    Compute average and max gap length from gap distribution dictionary
    
    Args:
        gap_distribution: Dictionary with gap lengths as keys and counts as values
        
    Returns:
        Dictionary containing average_gap_length and max_gap_length
    """
    total_gaps = 0
    total_length = 0
    max_length = 0
    
    for length_str, count_str in gap_distribution.items():
        length = int(length_str)
        count = int(count_str)
        
        total_gaps += count
        total_length += length * count
        max_length = max(max_length, length)
        
    avg_length = total_length / total_gaps if total_gaps > 0 else 0
    
    return {
        'avg_gap_length': avg_length,
        'max_gap_length': max_length,
        'total_gaps': total_gaps
    }




def compare_quality_reports(report1: Dict, report2: Dict) -> Dict:
    """Compare two data quality reports and highlight significant differences."""
    comparison: Dict[str,Any] = {
        'temporal_coverage': {
            report1["config"]["validation"].get("dataset_name"): {},
            report2["config"]["validation"].get("dataset_name"): {},
            # 'differences': {}
        },
        'data_completeness': {
            'gps': {},
            'accelerometer': {}
        },
        'data_quality': {
            'gps': {},
            'accelerometer': {}
        },
        'gap_analysis': {
            'gps': {},
            'accelerometer': {}
        },
        'accelerometer_features': {
            report1["config"]["validation"].get("dataset_name"): {},
            report2["config"]["validation"].get("dataset_name"): {}
        },
        'label_quality': {
            report1["config"]["validation"].get("dataset_name"): {},
            report2["config"]["validation"].get("dataset_name"): {}
        }
    }

    # 1. Temporal Coverage Analysis
    for dataset_idx, report in enumerate([report1, report2], 1):
        dataset_key = report["config"]["validation"].get("dataset_name")
        coverage = comparison['temporal_coverage'][dataset_key]
        
        coverage['study_duration_days'] = (pd.to_datetime(report['end_datetime']) - pd.to_datetime(report['start_datetime'])).days
        
        # GPS metrics
        n_gps_devices = len(report['gps']['devices'])
        coverage['gps'] = {
            'n_devices': n_gps_devices,
            'n_days': (pd.to_datetime(report['end_datetime']) - pd.to_datetime(report['start_datetime'])).days,
            'expected_records': int(report['gps']['total_expected_records']),
            'actual_records': int(report['gps']['total_final_records']),
            'coverage_pct': (int(report['gps']['total_final_records']) / int(report['gps']['total_expected_records'])) * 100,
            'avg_records_per_device_per_day': (int(report['gps']['total_final_records']) / 
                                             coverage['study_duration_days'] / 
                                             n_gps_devices)
        }

        # Accelerometer metrics
        n_acc_devices = len(report['accelerometer']['devices'])
        coverage['accelerometer'] = {
            'n_devices': n_acc_devices,
            'n_days': (pd.to_datetime(report['end_datetime']) - pd.to_datetime(report['start_datetime'])).days,
            'expected_records': int(report['accelerometer']['total_expected_records']),
            'actual_records': int(report['accelerometer']['total_final_records']),
            'coverage_pct': (int(report['accelerometer']['total_final_records']) / 
                           int(report['accelerometer']['total_expected_records'])) * 100,
            'avg_records_per_device_per_day': (int(report['accelerometer']['total_final_records']) / 
                                             coverage['study_duration_days'] / 
                                             n_acc_devices),
            'total_windows': int(report['accelerometer']['total_windows_computed']),
            'avg_windows_per_device_per_day': (int(report['accelerometer']['total_windows_computed']) / 
                                             coverage['study_duration_days'] / 
                                             n_acc_devices)
        }

    # 2. Data Quality Metrics
    for data_type in ['gps', 'accelerometer']:
        quality = comparison['data_quality'][data_type]
        
        for dataset_idx, report in enumerate([report1, report2], 1):
            dataset_key = report["config"]["validation"].get("dataset_name")
            
            # Common stats for both types
            dupes = []
            n_missing = []
            
            for device in report[data_type]['devices'].values():
                freq_stats = device.get('frequency_stats', {})
                dupes.append(int(freq_stats.get('duplicates', {}).get('n', 0)))
                n_missing.append(int(freq_stats.get('n_missing', 0)))

            base_metrics = {
                'total_duplicates': sum(dupes),
                'avg_duplicates_per_device': mean(dupes),
                'avg_duplicates_per_device_per_day': mean(dupes) / comparison['temporal_coverage'][dataset_key]['study_duration_days'],
                'total_missing': sum(n_missing),
                'avg_missing_per_day': mean(n_missing),
                'avg_missing_per_device_per_day': mean(n_missing) / comparison['temporal_coverage'][dataset_key]['study_duration_days']
            }

            if data_type == 'gps':
                n_zeros = []
                for device in report[data_type]['devices'].values():
                    zero_stats = device.get('zero_val_stats', {}).get('zero_coordinates', {})
                    n_zeros.append(int(zero_stats.get('total_zero_coords', 0)))

                quality[dataset_key] = {
                    **base_metrics,
                    'total_zeros': sum(n_zeros),
                    'avg_zeros_per_day': mean(n_zeros),
                    'avg_zeros_per_device_per_day': mean(n_zeros) / comparison['temporal_coverage'][dataset_key]['study_duration_days']
                }

            else:  # accelerometer
                n_interpolated = []
                global_gap_dict = {}
                
                for device in report[data_type]['devices'].values():
                    gap_stats = device.get('gap_stats', {})
                    n_interpolated.append(int(gap_stats.get('interpolated_gaps', 0)))
                    
                    gap_dist = gap_stats.get('gap_analysis', {}).get('gap_distribution', {})
                    for k, v in gap_dist.items():
                        k = int(k)
                        global_gap_dict[k] = global_gap_dict.get(k, 0) + int(v)

                quality[dataset_key] = {
                    **base_metrics,
                    'total_interpolated': sum(n_interpolated),
                    'avg_interpolated_per_day': mean(n_interpolated),
                    'avg_interpolated_per_device_per_day': mean(n_interpolated) / comparison['temporal_coverage'][dataset_key]['study_duration_days'],
                    'gap_dict': global_gap_dict
                }
    #3. 
    # First, update the gap analysis section in compare_quality_reports:
    # for data_type in ['gps', 'accelerometer']:
    # 3. Gap Analysis
    for data_type in ['gps', 'accelerometer']:
        gaps = comparison['gap_analysis'][data_type]
        
        for dataset_idx, report in enumerate([report1, report2], 1):
            dataset_key = report["config"]["validation"].get("dataset_name")
            global_gap_dist = {}
            all_device_gaps = []
            
            # First aggregate gaps across all devices
            for device in report[data_type]['devices'].values():
                if 'gap_analysis' in device['gap_stats']:
                    gap_stats = device['gap_stats']['gap_analysis']
                    gap_dist = gap_stats.get('gap_distribution', {})
                    
                    # Add this device's gaps to global distribution
                    for gap_size, count in gap_dist.items():
                        gap_size = int(gap_size)
                        global_gap_dist[gap_size] = global_gap_dist.get(gap_size, 0) + int(count)
                        # Add individual gaps to list for accurate statistics calculation
                        all_device_gaps.extend([gap_size] * int(count))
            
            # Calculate statistics from aggregated data
            total_gaps = len(all_device_gaps)
            if all_device_gaps:
                avg_gap_length = mean(all_device_gaps)
                median_gap_length = np.median(all_device_gaps)
                max_gap_length = max(all_device_gaps)
            else:
                avg_gap_length = median_gap_length = max_gap_length = 0
            
            # Store aggregated stats for this dataset
            gaps[dataset_key] = {
                'total_gaps': total_gaps,
                'avg_gap_length': avg_gap_length,
                'median_gap_length': median_gap_length,
                'max_gap_length': max_gap_length,
                'gap_distribution': global_gap_dist
            }

    # 4. Label Quality Analysis
    for dataset_idx, report in enumerate([report1, report2], 1):
        dataset_key = report["config"]["validation"].get("dataset_name")
        label_stats = comparison['label_quality'][dataset_key]
        
        if 'labels' in report:
            labels = report['labels']
            
            # Basic metrics
            label_stats.update({
                'total_records': int(labels.get('total_records', 0)),
                'unique_devices': int(labels.get('unique_devices', 0)),
                'activity_counts': labels.get('activity_counts', {}),
                'avg_records_per_device': (int(labels.get('total_records', 0)) / 
                                         int(labels.get('unique_devices', 1))),
                'study_duration_days': (pd.to_datetime(labels['overall_time_range']['end']) - 
                                      pd.to_datetime(labels['overall_time_range']['start'])).days,
            })
            
            # Processing stats with safe access
            processing_stats = {}
            if 'initial_shape' in labels:
                try:
                    processing_stats['initial_records'] = int(labels['initial_shape'].strip('()').split(',')[0])
                except (ValueError, IndexError):
                    processing_stats['initial_records'] = 0
            
            processing_stats.update({
                'final_records': int(labels.get('records_after_mapping', 0)),
                'tag_mapping_success_rate': float(labels.get('tag_id_mapping_success_rate', '0').strip('%'))
            })
            
            label_stats['processing_stats'] = processing_stats
            
            # Per-device statistics
            device_stats = {
                'min_records': float('inf'),
                'max_records': 0,
                'avg_records': 0,
                'activity_coverage': {}
            }
            
            devices = labels.get('devices', {})
            if devices:
                for device in devices.values():
                    n_records = int(device.get('records', 0))
                    device_stats['min_records'] = min(device_stats['min_records'], n_records)
                    device_stats['max_records'] = max(device_stats['max_records'], n_records)
                    
                    # Track activities per device
                    for activity, count in device.get('activity_counts', {}).items():
                        if activity not in device_stats['activity_coverage']:
                            device_stats['activity_coverage'][activity] = 0
                        device_stats['activity_coverage'][activity] += 1
                
                device_stats['avg_records'] = int(labels.get('total_records', 0)) / len(devices)
                
                # Convert activity coverage to percentages
                n_devices = len(devices)
                for activity in device_stats['activity_coverage']:
                    device_stats['activity_coverage'][activity] = (
                        device_stats['activity_coverage'][activity] / n_devices * 100
                    )
            else:
                device_stats['min_records'] = 0
                device_stats['max_records'] = 0
                device_stats['avg_records'] = 0
            
            label_stats['device_stats'] = device_stats



    # Add accelerometer feature comparison
    for dataset_idx, report in enumerate([report1, report2], 1):
        dataset_key = report["config"]["validation"].get("dataset_name")
        feature_stats = comparison['accelerometer_features'][dataset_key]
        
        # Aggregate feature statistics across all devices
        all_device_features = {}
        
        for device in report['accelerometer']['devices'].values():

            if 'computed_features' in device['acc_feature_stats']:
                for feature_name, stats in device['acc_feature_stats']['computed_features'].items():
                    if feature_name not in all_device_features:
                        all_device_features[feature_name] = {
                            'min': [],
                            'max': [],
                            'mean': [],
                            'count': []
                        }
                    
                    all_device_features[feature_name]['min'].append(float(stats['min']))
                    all_device_features[feature_name]['max'].append(float(stats['max']))
                    all_device_features[feature_name]['mean'].append(float(stats['mean']))
                    all_device_features[feature_name]['count'].append(int(stats['count']))
        
        # Calculate aggregate statistics
        for feature_name, stats in all_device_features.items():
            feature_stats[feature_name] = {
                'min': min(stats['min']),
                'max': max(stats['max']),
                'mean': sum(stats['mean']) / len(stats['mean']),
                'total_samples': sum(stats['count'])
            }

    return comparison



# Then, update the print_significant_differences function:
def print_significant_differences(comparison: Dict, k1, k2):
    """Print significant differences between datasets."""
    print("\nComparison Between Datasets:")
    
    print("\n1. TEMPORAL COVERAGE")
    for data_type in ['gps', 'accelerometer']:
        coverage1 = comparison['temporal_coverage'][k1][data_type]
        coverage2 = comparison['temporal_coverage'][k2][data_type]
        
        print(f"\n{data_type.upper()}")
        print(f"Number of devices:")
        print(f"  Dataset 1: {coverage1['n_devices']}")
        print(f"  Dataset 2: {coverage2['n_devices']}")

        print("Number of days")
        print(f"  Dataset 1: {coverage1["n_days"]}")
        print(f"  Dataset 2: {coverage2['n_days']}")
        

        print(f"\nCoverage %:")
        print(f"  Dataset 1: {coverage1['coverage_pct']:.1f}%")
        print(f"  Dataset 2: {coverage2['coverage_pct']:.1f}%")
        print(f"  Difference: {abs(coverage1['coverage_pct'] - coverage2['coverage_pct']):.1f}%")
        
        print(f"\nAverage records per device per day:")
        print(f"  Dataset 1: {coverage1['avg_records_per_device_per_day']:.1f}")
        print(f"  Dataset 2: {coverage2['avg_records_per_device_per_day']:.1f}")
        
        if data_type == 'accelerometer':
            print(f"\nAverage windows per device per day:")
            print(f"  Dataset 1: {coverage1['avg_windows_per_device_per_day']:.1f}")
            print(f"  Dataset 2: {coverage2['avg_windows_per_device_per_day']:.1f}")


    # 2. Data Quality
    print("\n2. DATA QUALITY")
    for data_type in ['gps', 'accelerometer']:
        quality1 = comparison['data_quality'][data_type][k1]
        quality2 = comparison['data_quality'][data_type][k2]
        
        print(f"\n{data_type.upper()}")
        print("Missing values per device per day:")
        print(f"  Dataset 1: {quality1['avg_missing_per_device_per_day']:.1f}")
        print(f"  Dataset 2: {quality2['avg_missing_per_device_per_day']:.1f}")
        
        print("\nDuplicates per device per day:")
        print(f"  Dataset 1: {quality1['avg_duplicates_per_device_per_day']:.1f}")
        print(f"  Dataset 2: {quality2['avg_duplicates_per_device_per_day']:.1f}")
        
        if data_type == 'gps':
            print("\nZero coordinates per device per day:")
            print(f"  Dataset 1: {quality1['avg_zeros_per_device_per_day']:.1f}")
            print(f"  Dataset 2: {quality2['avg_zeros_per_device_per_day']:.1f}")
        
        if data_type == 'accelerometer':
            print("\nInterpolated values per device per day:")
            print(f"  Dataset 1: {quality1['avg_interpolated_per_device_per_day']:.1f}")
            print(f"  Dataset 2: {quality2['avg_interpolated_per_device_per_day']:.1f}")

    print("\n3. GAP ANALYSIS")
    for data_type in ['gps', 'accelerometer']:
        gaps1 = comparison['gap_analysis'][data_type][k1]
        gaps2 = comparison['gap_analysis'][data_type][k2]
        
        print(f"\n{data_type.upper()}")
        print("Total gaps:")
        print(f"  Dataset 1: {gaps1['total_gaps']}")
        print(f"  Dataset 2: {gaps2['total_gaps']}")
        
        print("\nGap length statistics (seconds):")
        print(f"  Dataset 1:")
        print(f"    Mean: {gaps1['avg_gap_length']:.1f}")
        print(f"    Median: {gaps1['median_gap_length']:.1f}")
        print(f"    Maximum: {gaps1['max_gap_length']}")
        
        print(f"  Dataset 2:")
        print(f"    Mean: {gaps2['avg_gap_length']:.1f}")
        print(f"    Median: {gaps2['median_gap_length']:.1f}")
        print(f"    Maximum: {gaps2['max_gap_length']}")
        
        if data_type == 'accelerometer':
            print("\nGap Distribution:")
            gap_dist1 = gaps1.get('gap_distribution', {})
            gap_dist2 = gaps2.get('gap_distribution', {})
            
            all_lengths = sorted(set(list(gap_dist1.keys()) + list(gap_dist2.keys())))
            print("\nLength(s) | Dataset1 | Dataset2")
            print("-" * 30)
            for length in all_lengths:
                count1 = gap_dist1.get(int(length), 0)
                count2 = gap_dist2.get(int(length), 0)
                print(f"{length:>8} | {count1:>8} | {count2:>8}")

    # 4. Label Quality
    print("\n4. LABEL QUALITY")
    for dataset_idx in [k1, k2]:

        label_stats = comparison['label_quality'][dataset_idx]
        
        if label_stats:  # Only print if we have label data
            print(f"\nDataset {dataset_idx}:")
            print(f"  Total records: {label_stats['total_records']}")
            print(f"  Unique devices: {label_stats['unique_devices']}")
            print(f"  Study duration: {label_stats['study_duration_days']} days")
            print(f"  Average records per device: {label_stats['avg_records_per_device']:.1f}")
            
            print("\n  Activity counts:")
            for activity, count in label_stats['activity_counts'].items():
                print(f"    {activity}: {count}")
            
            print("\n  Device statistics:")
            device_stats = label_stats['device_stats']
            print(f"    Records per device: {device_stats['min_records']} - {device_stats['max_records']} (avg: {device_stats['avg_records']:.1f})")
            
            print("\n  Activity coverage (% of devices):")
            for activity, coverage in device_stats['activity_coverage'].items():
                print(f"    {activity}: {coverage:.1f}%")
            
            # print("\n  Processing statistics:")
            # proc_stats = label_stats['processing_stats']
            # print(f"    Initial records: {proc_stats['initial_records']}")
            # print(f"    Final records: {proc_stats['final_records']}")
            # print(f"    Tag mapping success rate: {proc_stats['tag_mapping_success_rate']}%")

    # Print significant differences
    if all(comparison['label_quality'].values()):
        print("\nSignificant Label Differences:")
        stats1 = comparison['label_quality'][k1]
        stats2 = comparison['label_quality'][k2]
        
        # Compare record counts
        record_diff = abs(stats1['total_records'] - stats2['total_records'])
        if record_diff > 100:
            print(f"\nTotal records difference: {record_diff}")
            
        # Compare activity distributions
        all_activities = set(stats1['activity_counts'].keys()) | set(stats2['activity_counts'].keys())
        for activity in all_activities:
            count1 = int(stats1['activity_counts'].get(activity, 0))
            count2 = int(stats2['activity_counts'].get(activity, 0))
            diff = abs(count1 - count2)
            if diff > 10:
                print(f"\n{activity} count difference: {diff}")
                print(f"  Dataset 1: {count1}")
                print(f"  Dataset 2: {count2}")


    # Add accelerometer feature comparison section
    print("\n5. ACCELEROMETER FEATURES")
    print("========================")
    
    features1 = comparison['accelerometer_features'][k1]
    features2 = comparison['accelerometer_features'][k2]
    
    # Get all unique feature names
    all_features = sorted(set(list(features1.keys()) + list(features2.keys())))
    
    for feature in all_features:
        print(f"\n{feature}:")
        if feature in features1 and feature in features2:
            print("  Dataset 1:")
            print(f"    Min: {features1[feature]['min']:.4f}")
            print(f"    Max: {features1[feature]['max']:.4f}")
            print(f"    Mean: {features1[feature]['mean']:.4f}")
            print(f"    Total samples: {features1[feature]['total_samples']}")
            
            print("  Dataset 2:")
            print(f"    Min: {features2[feature]['min']:.4f}")
            print(f"    Max: {features2[feature]['max']:.4f}")
            print(f"    Mean: {features2[feature]['mean']:.4f}")
            print(f"    Total samples: {features2[feature]['total_samples']}")
            
            # Calculate and print percent differences
            mean_diff_pct = abs(features1[feature]['mean'] - features2[feature]['mean']) / \
                          ((features1[feature]['mean'] + features2[feature]['mean']) / 2) * 100
            range_diff_pct = abs((features1[feature]['max'] - features1[feature]['min']) - \
                               (features2[feature]['max'] - features2[feature]['min'])) / \
                           ((features1[feature]['max'] - features1[feature]['min'] + \
                             features2[feature]['max'] - features2[feature]['min']) / 2) * 100
            
            print(f"  Differences:")
            print(f"    Mean difference: {mean_diff_pct:.1f}%")
            print(f"    Range difference: {range_diff_pct:.1f}%")



if __name__ == "__main__":
    with open('data\\processed\\RB_19\\all_cows_labeled.csv_dqr.json', 'r') as f1:
        RB19_Stats = json.load(f1)
    with open('data\\processed\\RB_22\\all_cows_labeled.csv_dqr.json', 'r') as f2:
        RB22_Stats = json.load(f2)
    
    comp = compare_quality_reports(RB19_Stats, RB22_Stats)
    k1 = RB19_Stats["config"]["validation"].get("dataset_name")
    k2 = RB22_Stats["config"]["validation"].get("dataset_name")
    print_significant_differences(comp, k1, k2)