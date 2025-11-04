# Demo Tables

Create demographics tables with summary statistics and statistical tests for research data.

## Installation

Install directly from GitHub:

    pip install git+https://github.com/yourusername/demo-tables.git

## Usage

    from demo_tables import fixed_demographics_table, fixed_grouped_demographics_table

    # Create simple demographics table
    table = fixed_demographics_table(df, var_list)

    # Create grouped demographics table with statistical tests
    grouped_table = fixed_grouped_demographics_table(df, var_list, group_var='treatment')

