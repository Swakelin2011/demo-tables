"""
Demo Tables Package
Create demographics tables with summary statistics and statistical tests
"""

from .demo_tables import (
    fixed_demographics_table,
    fixed_grouped_demographics_table,
    demographics_table_mean_sd,
    demographics_table_median_iqr,
    grouped_demographics_table_mean_sd,
    grouped_demographics_table_median_iqr,
    quick_fix_demographics
)

__version__ = "0.1.0"

__all__ = [
    'fixed_demographics_table',
    'fixed_grouped_demographics_table',
    'demographics_table_mean_sd',
    'demographics_table_median_iqr',
    'grouped_demographics_table_mean_sd',
    'grouped_demographics_table_median_iqr',
    'quick_fix_demographics'
]
