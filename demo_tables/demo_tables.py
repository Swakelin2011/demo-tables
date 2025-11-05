import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
import warnings

def fixed_demographics_table(df, var_list, categorical_vars=None, continuous_vars=None, 
                           ordinal_vars=None, decimal_places=1, include_missing=False, 
                           show_all_categories=True, reporting_format="auto"):
    """
    Create a demographics table with summary statistics and consistent formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    var_list : list
        List of variables to include in the table
    categorical_vars : list, optional
        Variables to force as categorical
    continuous_vars : list, optional
        Variables to force as continuous
    ordinal_vars : list, optional
        Variables to force as ordinal (will use median[IQR])
    decimal_places : int, default=1
        Number of decimal places for results
    include_missing : bool, default=False
        Whether to report missing data counts
    show_all_categories : bool, default=True
        Whether to show all categories for multi-categorical variables
    reporting_format : str, default="auto"
        Format for continuous variables: "auto", "mean_sd", or "median_iqr"
    
    Returns:
    --------
    pandas.DataFrame
        Demographics table with Variable and Result columns
    """
    
    if categorical_vars is None:
        categorical_vars = []
    if continuous_vars is None:
        continuous_vars = []
    if ordinal_vars is None:
        ordinal_vars = []
    
    results = []
    
    # Add overall N as first row
    overall_n = len(df)
    results.append({
        'Variable': 'Overall N',
        'Result': str(overall_n)
    })
    
    # Determine reporting formats for continuous variables
    variable_formats = {}
    
    for var in var_list:
        if var not in df.columns:
            continue
            
        var_type = _determine_variable_type(df, var, categorical_vars, continuous_vars, ordinal_vars)
        
        if var_type == "continuous":
            variable_formats[var] = _determine_reporting_format(df, var, reporting_format)
        elif var_type == "ordinal":
            variable_formats[var] = "median_iqr"  # Ordinal always uses median[IQR]
    
    # Process each variable
    for var in var_list:
        if var not in df.columns:
            print(f"Warning: '{var}' not found in dataframe")
            results.append({
                'Variable': var,
                'Result': 'Variable not found'
            })
            continue
            
        # Get non-missing data
        data_series = df[var]
        non_missing = data_series.dropna()
        n_valid = len(non_missing)
        n_missing = len(data_series) - n_valid
        
        # Skip if no valid data
        if n_valid == 0:
            results.append({
                'Variable': var,
                'Result': 'No valid data'
            })
            continue
        
        # Determine variable type
        var_type = _determine_variable_type(df, var, categorical_vars, continuous_vars, ordinal_vars)
        
        # Calculate statistics based on type
        if var_type == "categorical":
            result = _process_categorical_variable(
                non_missing, n_valid, n_missing, decimal_places, 
                include_missing, show_all_categories
            )
            # Replace PLACEHOLDER with actual variable name
            for r in result:
                if r['Variable'] == 'PLACEHOLDER':
                    r['Variable'] = var
            results.extend(result)
            
        elif var_type == "ordinal":
            result = _process_ordinal_variable(
                var, non_missing, n_valid, n_missing, decimal_places, include_missing
            )
            results.append(result)
            
        else:  # continuous
            result = _process_continuous_variable(
                var, non_missing, n_valid, n_missing, decimal_places,
                include_missing, variable_formats
            )
            results.append(result)
    
    return pd.DataFrame(results)


def fixed_grouped_demographics_table(df, var_list, group_var, categorical_vars=None, 
                                   continuous_vars=None, ordinal_vars=None, decimal_places=1, 
                                   include_tests=True, alpha=0.05, show_all_categories=True,
                                   include_missing=False, reporting_format="auto",
                                   mean_sd_vars=None, median_iqr_vars=None):
    """
    Create demographics table stratified by a grouping variable with CONSISTENT formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    var_list : list
        List of variables to include in the table
    group_var : str
        Variable to group by
    categorical_vars : list, optional
        Variables to force as categorical
    continuous_vars : list, optional
        Variables to force as continuous
    ordinal_vars : list, optional
        Variables to force as ordinal (will use median[IQR] and non-parametric tests)
    decimal_places : int, default=1
        Number of decimal places for results
    include_tests : bool, default=True
        Whether to include statistical tests
    alpha : float, default=0.05
        Significance level for tests
    show_all_categories : bool, default=True
        Whether to show all categories for multi-categorical variables
    include_missing : bool, default=False
        Whether to report missing data counts
    reporting_format : str, default="auto"
        Default format for continuous variables: "auto", "mean_sd", or "median_iqr"
    mean_sd_vars : list, optional
        Specific variables to force as mean±SD reporting
    median_iqr_vars : list, optional
        Specific variables to force as median[IQR] reporting
    
    Returns:
    --------
    pandas.DataFrame
        Grouped demographics table with Variable, group columns, and P-value
    """
    
    if categorical_vars is None:
        categorical_vars = []
    if continuous_vars is None:
        continuous_vars = []
    if ordinal_vars is None:
        ordinal_vars = []
    if mean_sd_vars is None:
        mean_sd_vars = []
    if median_iqr_vars is None:
        median_iqr_vars = []
    
    # Validate group variable
    if group_var not in df.columns:
        raise ValueError(f"Group variable '{group_var}' not found in dataframe")
    
    # Get unique groups, handle missing values properly
    group_data = df[group_var].dropna()
    if len(group_data) == 0:
        raise ValueError(f"No valid data in group variable '{group_var}'")
    
    groups = sorted(group_data.unique())
    
    # STEP 1: Analyze all variables to determine types and consistent formats
    variable_info = {}
    variable_formats = {}  # Store consistent format for each continuous/ordinal variable
    all_categories = {}
    
    for var in var_list:
        if var not in df.columns:
            variable_info[var] = {'type': 'missing', 'categories': []}
            continue
        
        # Get complete data for this variable
        var_data = df[var].dropna()
        
        if len(var_data) == 0:
            variable_info[var] = {'type': 'no_data', 'categories': []}
            continue
        
        # Determine variable type
        var_type = _determine_variable_type(df, var, categorical_vars, continuous_vars, ordinal_vars)
        variable_info[var] = {'type': var_type}
        
        # For continuous/ordinal variables, determine consistent reporting format
        if var_type == "continuous":
            variable_formats[var] = _determine_reporting_format_with_overrides(
                df, var, reporting_format, mean_sd_vars, median_iqr_vars
            )
        elif var_type == "ordinal":
            variable_formats[var] = "median_iqr"  # Ordinal always uses median[IQR]
        
        # For categorical variables, get all possible categories
        if var_type == "categorical":
            if _is_binary_numeric(var_data):
                variable_info[var]['categories'] = None
            else:
                categories = var_data.value_counts().index.tolist()
                all_categories[var] = categories
                variable_info[var]['categories'] = categories
        else:
            variable_info[var]['categories'] = None
    
    # STEP 2: Create individual tables for each group
    group_tables = {}
    
    for group in groups:
        group_df = df[df[group_var] == group].copy()
        
        # Create demographics table for this group
        group_table = _create_single_group_table_consistent(
            group_df, var_list, variable_info, variable_formats, all_categories,
            decimal_places, include_missing, show_all_categories
        )
        
        # Rename columns to include group identifier
        group_table = group_table.rename(columns={
            'Result': f'{group_var}_{group}'
        })
        
        group_tables[group] = group_table
    
    # STEP 3: Create unified structure across all groups
    all_variable_rows = set()
    for group_table in group_tables.values():
        all_variable_rows.update(group_table['Variable'].tolist())
    
    # Create ordered list of variables
    ordered_variables = _create_ordered_variable_list(var_list, all_variable_rows, variable_info)
    
    # Add overall N as first row if not already present
    if 'Overall N' not in ordered_variables:
        ordered_variables.insert(0, 'Overall N')
    
    # STEP 4: Build final aligned table
    final_table = pd.DataFrame({'Variable': ordered_variables})
    
    # Add columns for each group
    for group in groups:
        result_col = f'{group_var}_{group}'
        
        result_values = []
        
        for var in ordered_variables:
            if var == 'Overall N':
                # Add group size for overall N row
                group_size = len(df[df[group_var] == group])
                result_values.append(str(group_size))
            else:
                # Find matching row in this group's table
                matching_rows = group_tables[group][group_tables[group]['Variable'] == var]
                
                if not matching_rows.empty:
                    # Found exact match
                    result_values.append(matching_rows[result_col].iloc[0])
                else:
                    # No match - fill with appropriate defaults
                    if var.startswith('  '):  # Subcategory missing from this group
                        result_values.append("0 (0.0%)")
                    else:  # Main variable missing
                        result_values.append("No data")
        
        final_table[result_col] = result_values
    
    # STEP 5: Add statistical tests (only for non-Overall N rows)
    if include_tests:
        test_results = _perform_all_statistical_tests(
            df, final_table, group_var, variable_info, alpha
        )
        
        final_table['P-value'] = test_results['p_formatted']
        final_table['Significant'] = test_results['significant']
    
    return final_table


def _determine_variable_type(df, var, categorical_vars, continuous_vars, ordinal_vars):
    """Determine if variable is categorical, ordinal, or continuous"""
    
    if var in ordinal_vars:
        return "ordinal"
    elif var in categorical_vars:
        return "categorical"
    elif var in continuous_vars:
        return "continuous"
    else:
        # Auto-detect
        var_data = df[var].dropna()
        unique_vals = var_data.unique()
        is_numeric = pd.api.types.is_numeric_dtype(df[var])
        
        if is_numeric:
            # Check if it's actually discrete with few values
            if len(unique_vals) <= 10 and all(isinstance(x, (int, float)) and x == int(x) for x in unique_vals):
                return "categorical"
            else:
                return "continuous"
        else:
            return "categorical"


def _determine_reporting_format_with_overrides(df, var, reporting_format, mean_sd_vars, median_iqr_vars):
    """Determine consistent reporting format with variable-specific overrides"""
    
    # Check for specific variable overrides first
    if mean_sd_vars and var in mean_sd_vars:
        return "mean_sd"
    elif median_iqr_vars and var in median_iqr_vars:
        return "median_iqr"
    
    # Fall back to global format setting
    if reporting_format == "mean_sd":
        return "mean_sd"
    elif reporting_format == "median_iqr":
        return "median_iqr"
    else:  # "auto"
        # Decide based on overall distribution across all data
        try:
            var_data = df[var].dropna()
            mean_val = var_data.mean()
            std_val = var_data.std()
            median_val = var_data.median()
            
            # Use normality heuristic on overall data
            if std_val > 0 and abs(mean_val - median_val) / std_val < 0.5:
                return "mean_sd"
            else:
                return "median_iqr"
        except:
            return "mean_sd"  # default


def _determine_reporting_format(df, var, reporting_format):
    """Determine consistent reporting format for continuous variable (original function)"""
    
    if reporting_format == "mean_sd":
        return "mean_sd"
    elif reporting_format == "median_iqr":
        return "median_iqr"
    else:  # "auto"
        # Decide based on overall distribution across all data
        try:
            var_data = df[var].dropna()
            mean_val = var_data.mean()
            std_val = var_data.std()
            median_val = var_data.median()
            
            # Use normality heuristic on overall data
            if std_val > 0 and abs(mean_val - median_val) / std_val < 0.5:
                return "mean_sd"
            else:
                return "median_iqr"
        except:
            return "mean_sd"  # default


def _is_binary_numeric(data):
    """Check if data is binary 0/1"""
    return (pd.api.types.is_numeric_dtype(data) and 
            len(data.unique()) == 2 and 
            set(data.unique()) == {0, 1})


def _process_categorical_variable(non_missing, n_valid, n_missing, decimal_places, 
                                include_missing, show_all_categories):
    """Process a categorical variable and return results"""
    
    results = []
    
    # Check if binary 0/1
    if _is_binary_numeric(non_missing):
        # Binary numeric (0/1) - report count of 1s
        count = int(non_missing.sum())
        percent = (count / n_valid) * 100 if n_valid > 0 else 0
        result = f"{count} ({percent:.{decimal_places}f}%)"
        
        if include_missing and n_missing > 0:
            result += f" (Missing: {n_missing})"
        
        results.append({
            'Variable': 'PLACEHOLDER',
            'Result': result
        })
    else:
        # Multi-category variable
        value_counts = non_missing.value_counts()
        
        if show_all_categories and len(value_counts) > 1:
            # Main variable row
            main_result = ""
            if include_missing and n_missing > 0:
                main_result = f"(Missing: {n_missing})"
            
            results.append({
                'Variable': 'PLACEHOLDER',
                'Result': main_result
            })
            
            # Subcategory rows
            for category, count in value_counts.items():
                percent = (count / n_valid) * 100
                category_clean = str(category).strip()
                
                results.append({
                    'Variable': f"  {category_clean}",
                    'Result': f"{count} ({percent:.{decimal_places}f}%)"
                })
        else:
            # Show only most frequent
            if len(value_counts) > 0:
                most_frequent = value_counts.index[0]
                count = value_counts.iloc[0]
                percent = (count / n_valid) * 100
                result = f"{most_frequent}: {count} ({percent:.{decimal_places}f}%)"
                
                if include_missing and n_missing > 0:
                    result += f" (Missing: {n_missing})"
                
                results.append({
                    'Variable': 'PLACEHOLDER',
                    'Result': result
                })
    
    return results


def _process_ordinal_variable(var, non_missing, n_valid, n_missing, decimal_places, include_missing):
    """Process an ordinal variable using median[IQR]"""
    
    try:
        median_val = non_missing.median()
        q25 = non_missing.quantile(0.25)
        q75 = non_missing.quantile(0.75)
        result = f"{median_val:.{decimal_places}f} [{q25:.{decimal_places}f}, {q75:.{decimal_places}f}]"
        
        if include_missing and n_missing > 0:
            result += f" (Missing: {n_missing})"
        
        return {
            'Variable': var,
            'Result': result
        }
        
    except Exception as e:
        return {
            'Variable': var,
            'Result': f"Error: {str(e)}"
        }


def _process_continuous_variable(var, non_missing, n_valid, n_missing, decimal_places,
                               include_missing, variable_formats):
    """Process a continuous variable with consistent formatting"""
    
    try:
        format_type = variable_formats.get(var, "mean_sd")
        
        if format_type == "mean_sd":
            mean_val = non_missing.mean()
            std_val = non_missing.std()
            result = f"{mean_val:.{decimal_places}f} ± {std_val:.{decimal_places}f}"
        else:  # median_iqr
            median_val = non_missing.median()
            q25 = non_missing.quantile(0.25)
            q75 = non_missing.quantile(0.75)
            result = f"{median_val:.{decimal_places}f} [{q25:.{decimal_places}f}, {q75:.{decimal_places}f}]"
        
        if include_missing and n_missing > 0:
            result += f" (Missing: {n_missing})"
        
        return {
            'Variable': var,
            'Result': result
        }
        
    except Exception as e:
        return {
            'Variable': var,
            'Result': f"Error: {str(e)}"
        }


def _create_single_group_table_consistent(df, var_list, variable_info, variable_formats, all_categories,
                                        decimal_places, include_missing, show_all_categories):
    """Create demographics table for single group with consistent formatting"""
    
    results = []
    
    # Add overall N for this group
    overall_n = len(df)
    results.append({
        'Variable': 'Overall N',
        'Result': str(overall_n)
    })
    
    for var in var_list:
        if var not in df.columns:
            results.append({
                'Variable': var,
                'Result': 'Variable not found'
            })
            if var in all_categories:
                for category in all_categories[var]:
                    results.append({
                        'Variable': f"  {str(category).strip()}",
                        'Result': "0 (0.0%)"
                    })
            continue
        
        # Get data for this variable in this group
        var_data = df[var].dropna()
        n_valid = len(var_data)
        n_total = len(df[var])
        n_missing = n_total - n_valid
        
        var_info = variable_info.get(var, {})
        var_type = var_info.get('type', 'unknown')
        
        if n_valid == 0:
            main_result = "No data"
            if include_missing and n_missing > 0:
                main_result += f" (Missing: {n_missing})"
            
            results.append({
                'Variable': var,
                'Result': main_result
            })
            
            if var in all_categories:
                for category in all_categories[var]:
                    results.append({
                        'Variable': f"  {str(category).strip()}",
                        'Result': "0 (0.0%)"
                    })
            continue
        
        # Process based on variable type
        if var_type == "categorical":
            if _is_binary_numeric(var_data):
                count = int(var_data.sum())
                percent = (count / n_valid) * 100 if n_valid > 0 else 0
                result = f"{count} ({percent:.{decimal_places}f}%)"
                
                if include_missing and n_missing > 0:
                    result += f" (Missing: {n_missing})"
                
                results.append({
                    'Variable': var,
                    'Result': result
                })
            
            else:
                if show_all_categories and var in all_categories:
                    main_result = ""
                    if include_missing and n_missing > 0:
                        main_result = f"(Missing: {n_missing})"
                    
                    results.append({
                        'Variable': var,
                        'Result': main_result
                    })
                    
                    value_counts = var_data.value_counts()
                    
                    for category in all_categories[var]:
                        count = value_counts.get(category, 0)
                        percent = (count / n_valid) * 100 if n_valid > 0 else 0
                        
                        results.append({
                            'Variable': f"  {str(category).strip()}",
                            'Result': f"{count} ({percent:.{decimal_places}f}%)"
                        })
                
                else:
                    value_counts = var_data.value_counts()
                    if len(value_counts) > 0:
                        most_frequent = value_counts.index[0]
                        count = value_counts.iloc[0]
                        percent = (count / n_valid) * 100
                        result = f"{most_frequent}: {count} ({percent:.{decimal_places}f}%)"
                        
                        if include_missing and n_missing > 0:
                            result += f" (Missing: {n_missing})"
                        
                        results.append({
                            'Variable': var,
                            'Result': result
                        })
        
        elif var_type == "ordinal":
            # Ordinal variables always use median[IQR]
            try:
                median_val = var_data.median()
                q25 = var_data.quantile(0.25)
                q75 = var_data.quantile(0.75)
                result = f"{median_val:.{decimal_places}f} [{q25:.{decimal_places}f}, {q75:.{decimal_places}f}]"
                
                if include_missing and n_missing > 0:
                    result += f" (Missing: {n_missing})"
                
                results.append({
                    'Variable': var,
                    'Result': result
                })
            
            except Exception as e:
                results.append({
                    'Variable': var,
                    'Result': f"Error: {str(e)}"
                })
        
        elif var_type == "continuous":
            try:
                format_type = variable_formats.get(var, "mean_sd")
                
                if format_type == "mean_sd":
                    mean_val = var_data.mean()
                    std_val = var_data.std()
                    result = f"{mean_val:.{decimal_places}f} ± {std_val:.{decimal_places}f}"
                
                else:  # median_iqr
                    median_val = var_data.median()
                    q25 = var_data.quantile(0.25)
                    q75 = var_data.quantile(0.75)
                    result = f"{median_val:.{decimal_places}f} [{q25:.{decimal_places}f}, {q75:.{decimal_places}f}]"
                
                if include_missing and n_missing > 0:
                    result += f" (Missing: {n_missing})"
                
                results.append({
                    'Variable': var,
                    'Result': result
                })
            
            except Exception as e:
                results.append({
                    'Variable': var,
                    'Result': f"Error: {str(e)}"
                })
        
        else:
            results.append({
                'Variable': var,
                'Result': "Unknown variable type"
            })
    
    return pd.DataFrame(results)


def _create_ordered_variable_list(var_list, all_variable_rows, variable_info):
    """Create consistently ordered list of variables and subcategories without duplication"""
    
    ordered_variables = []

    for var in var_list:
        if var in all_variable_rows:
            ordered_variables.append(var)
        
        if var in variable_info and variable_info[var].get('categories'):
            seen = set()
            for cat in variable_info[var]['categories']:
                subcat = f"  {str(cat).strip()}"
                if subcat in all_variable_rows and subcat not in seen:
                    ordered_variables.append(subcat)
                    seen.add(subcat)
    
    return ordered_variables


def _perform_all_statistical_tests(df, final_table, group_var, variable_info, alpha):
    """Perform statistical tests for all main variables"""
    
    p_formatted = []
    significant = []
    
    for _, row in final_table.iterrows():
        var = row['Variable']
        
        if var == 'Overall N' or var.startswith('  '):
            p_formatted.append("")
            significant.append("")
        else:
            if var not in df.columns:
                p_formatted.append("N/A")
                significant.append("")
                continue
            
            var_info = variable_info.get(var, {})
            var_type = var_info.get('type', 'unknown')
            
            if var_type in ['missing', 'no_data', 'unknown']:
                p_formatted.append("N/A")
                significant.append("")
                continue
            
            # Perform the statistical test
            test_name, p_value = _perform_single_statistical_test(df, var, group_var, var_type)
            
            # Format p-value
            if pd.isna(p_value):
                p_formatted.append("N/A")
            elif p_value < 0.001:
                p_formatted.append("<0.001")
            else:
                p_formatted.append(f"{p_value:.3f}")
            
            # Significance
            if pd.isna(p_value):
                significant.append("")
            elif p_value < alpha:
                significant.append("*")
            else:
                significant.append("")
    
    return {
        'p_formatted': p_formatted,
        'significant': significant
    }


def _perform_single_statistical_test(df, var, group_var, var_type):
    """Perform appropriate statistical test for a single variable"""
    
    if var not in df.columns:
        return "Variable not found", np.nan
    
    if group_var not in df.columns:
        return "Group variable not found", np.nan
    
    # Get data with both variables present
    test_df = df[[var, group_var]].copy()
    
    # Handle group variable
    if test_df[group_var].dtype == 'object':
        try:
            numeric_converted = pd.to_numeric(test_df[group_var], errors='coerce')
            non_null_original = test_df[group_var].notna().sum()
            non_null_converted = numeric_converted.notna().sum()
            
            if non_null_converted < 0.9 * non_null_original:
                pass
            else:
                test_df[group_var] = numeric_converted
        except:
            pass
    
    test_df = test_df.dropna(subset=[group_var])
    
    if len(test_df) == 0:
        return f"No observations with group information", np.nan
    
    groups = test_df[group_var].unique()
    groups = groups[pd.notna(groups)]
    
    if len(groups) < 2:
        return f"Only {len(groups)} group(s) found", np.nan
    
    test_df = test_df.dropna(subset=[var])
    
    if len(test_df) < 2:
        return f"Insufficient data after removing missing values", np.nan
    
    remaining_groups = test_df[group_var].unique()
    remaining_groups = remaining_groups[pd.notna(remaining_groups)]
    
    if len(remaining_groups) < 2:
        return f"Only {len(remaining_groups)} group(s) remaining", np.nan
    
    try:
        if var_type == "categorical":
            return _perform_categorical_test(test_df, var, group_var)
        elif var_type == "ordinal":
            return _perform_ordinal_test(test_df, var, group_var)
        else:  # continuous variable
            return _perform_continuous_test(test_df, var, group_var)
    
    except Exception as e:
        return f"Test error: {str(e)}", np.nan


def _perform_categorical_test(test_df, var, group_var):
    """Perform statistical test for categorical variables"""
    
    try:
        contingency_table = pd.crosstab(test_df[var], test_df[group_var])
    except Exception as e:
        return f"Failed to create contingency table: {str(e)}", np.nan
    
    if contingency_table.size == 0 or contingency_table.sum().sum() == 0:
        return "Empty contingency table", np.nan
    
    try:
        expected = stats.contingency.expected_freq(contingency_table)
        min_expected = expected.min()
        
        if contingency_table.shape == (2, 2) and min_expected < 5:
            try:
                table_array = contingency_table.values
                odds_ratio, p_value = fisher_exact(table_array)
                return "Fisher's exact", p_value
            except Exception as e:
                return f"Fisher's exact test failed: {str(e)}", np.nan
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            return "Chi-square", p_value
        except Exception as e:
            return f"Chi-square test failed: {str(e)}", np.nan
            
    except Exception as e:
        return f"Failed to calculate expected frequencies: {str(e)}", np.nan


def _perform_ordinal_test(test_df, var, group_var):
    """Perform non-parametric statistical test for ordinal variables"""
    
    groups = test_df[group_var].unique()
    groups = sorted([g for g in groups if pd.notna(g)])
    
    if len(groups) < 2:
        return "Insufficient groups for ordinal test", np.nan
    
    # Get data for each group
    group_data = []
    for group in groups:
        group_vals = test_df[test_df[group_var] == group][var].values
        group_vals = group_vals[~pd.isna(group_vals)]
        if len(group_vals) > 0:
            group_data.append(group_vals)
    
    if len(group_data) < 2:
        return "Insufficient groups with data", np.nan
    
    group_sizes = [len(group) for group in group_data]
    min_group_size = min(group_sizes)
    
    if min_group_size < 2:
        return f"Group(s) too small", np.nan
    
    try:
        if len(groups) == 2:
            # Mann-Whitney U test for two groups
            group1, group2 = group_data[0], group_data[1]
            
            try:
                u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                
                if pd.isna(p_value):
                    return "Mann-Whitney U test returned NaN", np.nan
                
                return "Mann-Whitney U", p_value
                
            except Exception as e:
                return f"Mann-Whitney U test failed: {str(e)}", np.nan
        
        else:
            # Kruskal-Wallis test for more than two groups
            try:
                h_stat, p_value = kruskal(*group_data)
                
                if pd.isna(p_value):
                    return "Kruskal-Wallis test returned NaN", np.nan
                
                return "Kruskal-Wallis", p_value
                
            except Exception as e:
                return f"Kruskal-Wallis test failed: {str(e)}", np.nan
    
    except Exception as e:
        return f"Ordinal test error: {str(e)}", np.nan


def _perform_continuous_test(test_df, var, group_var):
    """Perform statistical test for continuous variables"""
    
    groups = test_df[group_var].unique()
    groups = sorted([g for g in groups if pd.notna(g)])
    
    if len(groups) < 2:
        return "Insufficient groups for continuous test", np.nan
    
    # Get data for each group
    group_data = []
    for group in groups:
        group_vals = test_df[test_df[group_var] == group][var].values
        group_vals = group_vals[~pd.isna(group_vals)]
        if len(group_vals) > 0:
            group_data.append(group_vals)
    
    if len(group_data) < 2:
        return "Insufficient groups with data", np.nan
    
    group_sizes = [len(group) for group in group_data]
    min_group_size = min(group_sizes)
    
    if min_group_size < 2:
        return f"Group(s) too small", np.nan
    
    try:
        if len(groups) == 2:
            group1, group2 = group_data[0], group_data[1]
            
            try:
                levene_stat, levene_p = stats.levene(group1, group2)
                equal_var = levene_p > 0.05
            except:
                equal_var = True
            
            try:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
                
                if pd.isna(p_value):
                    return "t-test returned NaN", np.nan
                
                return "t-test", p_value
                
            except Exception as e:
                return f"t-test failed: {str(e)}", np.nan
        
        else:
            try:
                f_stat, p_value = stats.f_oneway(*group_data)
                
                if pd.isna(p_value):
                    return "ANOVA returned NaN", np.nan
                
                return "ANOVA", p_value
                
            except Exception as e:
                return f"ANOVA failed: {str(e)}", np.nan
    
    except Exception as e:
        return f"Continuous test error: {str(e)}", np.nan


# Helper functions for checking and fixing duplicates
def check_table_duplicates(table, table_name):
    """Check a table for duplicate rows"""
    print(f"\n=== {table_name} ===")
    print(f"Total rows: {len(table)}")
    
    main_vars = table[~table['Variable'].str.startswith('  ')]['Variable'].tolist()
    duplicate_vars = [var for var in set(main_vars) if main_vars.count(var) > 1]
    
    if duplicate_vars:
        print(f"❌ DUPLICATES FOUND: {duplicate_vars}")
        for dup_var in duplicate_vars:
            dup_rows = table[table['Variable'] == dup_var]
            print(f"  '{dup_var}' appears {len(dup_rows)} times at rows: {dup_rows.index.tolist()}")
    else:
        print("✅ No main variable duplicates found")
    
    subcat_rows = table[table['Variable'].str.startswith('  ')]
    if len(subcat_rows) > 0:
        duplicate_subcats = subcat_rows[subcat_rows.duplicated(['Variable'], keep=False)]
        if len(duplicate_subcats) > 0:
            print(f"❌ DUPLICATE SUBCATEGORIES: {len(duplicate_subcats)} rows")
            for var in duplicate_subcats['Variable'].unique():
                print(f"  '{var}' appears multiple times")
        else:
            print("✅ No subcategory duplicates found")


def remove_duplicate_subcategories(table):
    """Remove duplicate subcategory rows from an existing table"""
    print(f"Original table shape: {table.shape}")
    
    main_vars = table[~table['Variable'].str.startswith('  ')].copy()
    subcats = table[table['Variable'].str.startswith('  ')].copy()
    
    print(f"Main variables: {len(main_vars)}")
    print(f"Subcategories before dedup: {len(subcats)}")
    
    subcats_dedup = subcats.drop_duplicates(subset=['Variable'], keep='first')
    
    print(f"Subcategories after dedup: {len(subcats_dedup)}")
    
    result = pd.concat([main_vars, subcats_dedup], ignore_index=True)
    result = result.sort_index()
    
    print(f"Final table shape: {result.shape}")
    return result


# Convenience functions for common use cases
def demographics_table_mean_sd(df, var_list, **kwargs):
    """Create demographics table with mean±SD for all continuous variables"""
    return fixed_demographics_table(df, var_list, reporting_format="mean_sd", **kwargs)


def demographics_table_median_iqr(df, var_list, **kwargs):
    """Create demographics table with median[IQR] for all continuous variables"""
    return fixed_demographics_table(df, var_list, reporting_format="median_iqr", **kwargs)


def grouped_demographics_table_mean_sd(df, var_list, group_var, **kwargs):
    """Create grouped demographics table with mean±SD for all continuous variables"""
    return fixed_grouped_demographics_table(df, var_list, group_var, reporting_format="mean_sd", **kwargs)


def grouped_demographics_table_median_iqr(df, var_list, group_var, **kwargs):
    """Create grouped demographics table with median[IQR] for all continuous variables"""
    return fixed_grouped_demographics_table(df, var_list, group_var, reporting_format="median_iqr", **kwargs)


def quick_fix_demographics(df, var_list, group_var, force_format="mean_sd", 
                          mean_sd_vars=None, median_iqr_vars=None, **kwargs):
    """
    Quick fix function that ensures 100% consistent reporting
    
    Parameters:
    -----------
    force_format : str, default="mean_sd"
        "mean_sd" - Force mean±SD for ALL continuous variables
        "median_iqr" - Force median[IQR] for ALL continuous variables
        "auto" - Auto-decide but allow overrides
    mean_sd_vars : list, optional
        Specific variables to force as mean±SD reporting
    median_iqr_vars : list, optional
        Specific variables to force as median[IQR] reporting
    """
    
    return fixed_grouped_demographics_table(
        df, var_list, group_var, 
        reporting_format=force_format,
        mean_sd_vars=mean_sd_vars,
        median_iqr_vars=median_iqr_vars,
        **kwargs
    )


# Example usage and testing functions
def example_usage():
    """Example usage of the demographics functions with ordinal variables"""
    
    np.random.seed(42)
    n = 200
    
    sample_df = pd.DataFrame({
        'age': np.random.normal(65, 15, n),
        'bmi': np.random.normal(28, 5, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'diabetes': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'treatment': np.random.choice(['A', 'B'], n),
        'pain_score': np.random.exponential(2, n),
        'satisfaction': np.random.choice(['Poor', 'Good', 'Excellent'], n, p=[0.2, 0.5, 0.3]),
        'pain_level': np.random.choice([1, 2, 3, 4, 5], n),  # Ordinal: 1-5 scale
        'education': np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])  # Ordinal: education level
    })
    
    # Variables to analyze
    variables = ['age', 'bmi', 'gender', 'diabetes', 'pain_score', 'satisfaction', 'pain_level', 'education']
    
    print("=== EXAMPLE 1: Simple Demographics Table with Ordinal ===")
    
    table1 = fixed_demographics_table(
        sample_df, variables, 
        ordinal_vars=['pain_level', 'education'],
        decimal_places=1
    )
    print(table1)
    print()
    
    print("=== EXAMPLE 2: Grouped Demographics Table with Ordinal ===")
    
    grouped_table = fixed_grouped_demographics_table(
        sample_df, variables, 'treatment',
        ordinal_vars=['pain_level', 'education'],
        decimal_places=1, 
        include_tests=True
    )
    print(grouped_table)
    print()
    
    print("=== EXAMPLE 3: Mixed Format with Ordinal ===")
    
    grouped_table2 = fixed_grouped_demographics_table(
        sample_df, variables, 'treatment',
        ordinal_vars=['pain_level', 'education'],
        reporting_format="auto",
        mean_sd_vars=['age', 'bmi'],
        median_iqr_vars=['pain_score'],
        decimal_places=1,
        include_tests=True
    )
    print(grouped_table2)
    print()
    
    return sample_df, grouped_table2


def validate_consistency(table, continuous_vars):
    """Validate that continuous variables have consistent formatting across groups"""
    print("=== CONSISTENCY VALIDATION ===")
    
    group_cols = [col for col in table.columns if col not in ['Variable', 'P-value', 'Significant']]
    
    inconsistent_vars = []
    
    for _, row in table.iterrows():
        var = row['Variable']
        
        if var == 'Overall N' or var.startswith('  '):
            continue
            
        var_base = var.replace('preop ', '').replace('postop ', '').replace(' (4 weeks)', '').replace(' (6-weeks)', '')
        
        if any(cont_var in var.lower() for cont_var in continuous_vars):
            formats = []
            for col in group_cols:
                value = str(row[col])
                if '±' in value:
                    formats.append('mean_sd')
                elif '[' in value and ']' in value:
                    formats.append('median_iqr')
                else:
                    formats.append('other')
            
            unique_formats = set(formats)
            if len(unique_formats) > 1:
                inconsistent_vars.append({
                    'variable': var,
                    'formats': dict(zip(group_cols, formats))
                })
                print(f"❌ INCONSISTENT: {var}")
                for col, fmt in zip(group_cols, formats):
                    print(f"  {col}: {fmt} ({row[col]})")
            else:
                print(f"✅ CONSISTENT: {var} (all {list(unique_formats)[0]})")
    
    if inconsistent_vars:
        print(f"\n❌ Found {len(inconsistent_vars)} inconsistent variables")
        return False
    else:
        print(f"\n✅ All continuous variables have consistent formatting!")
        return True

def create_odds_ratio_table(model, alpha=0.05):
    """
    Creates an odds ratio table from a fitted logistic regression model
    
    Parameters:
    -----------
    model : fitted statsmodels logistic regression model
        The fitted model from statsmodels (e.g., smf.logit().fit())
    alpha : float, default=0.05
        Significance level for confidence intervals (default 0.05 for 95% CI)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: Coefficient, Odds Ratio, Lower CI, Upper CI, P-value, Sig
    
    Example:
    --------
    >>> model = smf.logit('outcome ~ var1 + var2', data=df).fit()
    >>> or_table = create_odds_ratio_table(model)
    >>> print(or_table)
    """
    import pandas as pd
    import numpy as np
    
    # Get coefficients and confidence intervals
    params = model.params
    conf = model.conf_int(alpha=alpha)
    conf.columns = ['Lower CI', 'Upper CI']
    
    # Calculate odds ratios
    odds_ratios = np.exp(params)
    or_conf = np.exp(conf)
    
    # Get p-values
    pvalues = model.pvalues
    
    # Create the table
    or_table = pd.DataFrame({
        'Coefficient': params,
        'Odds Ratio': odds_ratios,
        'Lower CI': or_conf['Lower CI'],
        'Upper CI': or_conf['Upper CI'],
        'P-value': pvalues
    })
    
    # Round appropriately
    or_table['Coefficient'] = or_table['Coefficient'].round(4)
    or_table['Odds Ratio'] = or_table['Odds Ratio'].round(4)
    or_table['Lower CI'] = or_table['Lower CI'].round(4)
    or_table['Upper CI'] = or_table['Upper CI'].round(4)
    or_table['P-value'] = or_table['P-value'].round(4)
    
    # Add significance stars
    or_table['Sig'] = or_table['P-value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
    )
    
    return or_table