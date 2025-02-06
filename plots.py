#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:38:29 2024

@author: michelev
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import recfunctions

#%%


def calculate_avg_std_df_with_cubic(matches_df, agent_df, partner_df, agent_id_col, partner_id_col):
    """
    Calculate average and standard deviation of matched partner valuations using DataFrames,
    and fit a cubic polynomial to the averages.

    Parameters:
    - matches_df: DataFrame with matches, containing agent and partner IDs.
    - agent_df: DataFrame with agent IDs and their valuations.
    - partner_df: DataFrame with partner IDs and their valuations.
    - agent_id_col: Column name for agent IDs in matches_df.
    - partner_id_col: Column name for partner IDs in matches_df.

    Returns:
    - summary: DataFrame with agent valuations, average partner valuations, and standard deviations.
    - cubic_fit: Cubic polynomial coefficients for the fit to AvgPartnerVal as a function of AgentVal.
    """
    # Merge matches with partner valuations
    merged_df = matches_df.merge(partner_df, left_on=partner_id_col, right_on='ID', how='left')
    
    # Merge with agent valuations
    merged_df = merged_df.merge(agent_df, left_on=agent_id_col, right_on='ID', how='left', suffixes=('_partner', '_agent'))
    
    #print(merged_df.columns)
    #merged_df = merged_df.loc[merged_df['AvgVal_agent']>=-0.05,:]
    # Group by agent ID and compute average and std of partner valuations
    summary = merged_df.groupby(agent_id_col).agg(
        AgentVal=('AvgVal_agent', 'first'),  # Take the agent's valuation
        CountPartners=('AvgVal_partner', 'count'),
        AvgPartnerVal=('AvgVal_partner', 'mean'),
        StdPartnerVal=('AvgVal_partner', 'std')
    ).reset_index()

    # Fit a cubic polynomial to the averages
    if len(summary) > 3:  # Ensure we have enough data points for a cubic fit
        cubic_fit = np.polyfit(summary['AgentVal'], summary['AvgPartnerVal'], 3)
        #cubic_fit = np.polyfit(summary['AgentVal'], summary['CountPartners'], 2)
    else:
        cubic_fit = None  # Not enough data points to perform a cubic fit

    return summary, cubic_fit

def calculate_avg_std_df(matches_df, agent_df, partner_df, agent_id_col, partner_id_col):
    """
    Calculate average and standard deviation of matched partner valuations using DataFrames.

    Parameters:
    - matches_df: DataFrame with matches, containing agent and partner IDs.
    - agent_df: DataFrame with agent IDs and their valuations.
    - partner_df: DataFrame with partner IDs and their valuations.
    - agent_id_col: Column name for agent IDs in matches_df.
    - partner_id_col: Column name for partner IDs in matches_df.

    Returns:
    - DataFrame with agent valuations, average partner valuations, and standard deviations.
    """
    # Merge matches with partner valuations
    merged_df = matches_df.merge(partner_df, left_on=partner_id_col, right_on='ID', how='left')
    
    # Merge with agent valuations
    merged_df = merged_df.merge(agent_df, left_on=agent_id_col, right_on='ID', how='left', suffixes=('_partner', '_agent'))

    # Group by agent ID and compute average and std of partner valuations
    summary = merged_df.groupby(agent_id_col).agg(
        AgentVal=('Valuation_agent', 'first'),  # Take the agent's valuation
        CountPartners=('Valuation_partner', 'count'),
        AvgPartnerVal=('Valuation_partner', 'mean'),
        StdPartnerVal=('Valuation_partner', 'std')
    ).reset_index()

    return summary

def compute_and_plot_mean_valuation(sim, matches_df, beta_vector_buyer, beta_vector_seller, market_ids = None):
    """
    Compute mean_valuation for both empirical and theoretical distributions using predictors and beta vectors,
    and return the plots for buyers and sellers.

    Parameters:
    - sim: Simulation object containing buyers and sellers data and theoretical matches.
    - matches_df: DataFrame with 'buyer_ID' and 'seller_ID' for empirical matches.
    - beta_vector_buyer: Array-like, beta coefficients for buyers.
    - beta_vector_seller: Array-like, beta coefficients for sellers.

    Returns:
    - seller_plot (matplotlib.figure.Figure): Figure object for the sellers' plot.
    - buyer_plot (matplotlib.figure.Figure): Figure object for the buyers' plot.
    """
    # Compute predicted valuations using beta vectors
    if ~isinstance(sim, dict):
        sim_buyers_data = recfunctions.stack_arrays(
            [sim[market_id].buyers.data for market_id in market_ids],
            usemask=False,
            asrecarray=True
        )
    else:
        sim_buyers_data = sim.buyers.data
        #buyers_predictors = np.concatenate([sims[market_id].buyers.data['Predictors'] for market_id in market_ids], axis=0)
    if ~isinstance(sim, dict):
       sim_sellers_data = recfunctions.stack_arrays(
           [sim[market_id].sellers.data for market_id in market_ids],
           usemask=False,
           asrecarray=True
       )
    else:
        sim_sellers_data = sim.sellers.data
        #sellers_predictors = np.concatenate([sim[market_id].sellers.data['Predictors'] for market_id in market_ids], axis=0)

    if ~isinstance(sim, dict):
        sim_buyers_matches = {}
        for d in (sim[market_id].buyers_matches for market_id in market_ids):
            sim_buyers_matches = {**sim_buyers_matches, **d}
    else:
        sim_buyers_matches = sim.buyers_matches
        #buyers_predictors = np.concatenate([sims[market_id].buyers.data['Predictors'] for market_id in market_ids], axis=0)
    if ~isinstance(sim, dict):
       sim_sellers_matches = {}
       for d in (sim[market_id].sellers_matches for market_id in market_ids):
           sim_sellers_matches = {**sim_sellers_matches, **d}
       
    else:
        sim_sellers_matches = sim.sellers_matches
        
    buyers_predictors = sim_buyers_data['Predictors']  # Predictors for buyers
    sellers_predictors = sim_sellers_data['Predictors']  # Predictors for sellers
    
    buyers_val_pred = np.dot(buyers_predictors, beta_vector_buyer) / 100
    sellers_val_pred = np.dot(sellers_predictors, beta_vector_seller) / 100

    # Add AvgVal to buyers and sellers data
    buyers_data = recfunctions.append_fields(
        sim_buyers_data, 'AvgVal', buyers_val_pred, usemask=False
    )
    sellers_data = recfunctions.append_fields(
        sim_sellers_data, 'AvgVal', sellers_val_pred, usemask=False
    )

    # Convert to DataFrames
    buyers_df = pd.DataFrame({name: buyers_data[name] for name in buyers_data.dtype.names if name != 'Predictors'})
    sellers_df = pd.DataFrame({name: sellers_data[name] for name in sellers_data.dtype.names if name != 'Predictors'})

    # Theoretical matches
    theoretical_matches_buyers = {buyer_id: set(seller_ids) 
                                   for buyer_id, seller_ids in sim_buyers_matches.items()}
    theoretical_matches_sellers = {seller_id: set(buyer_ids) 
                                    for seller_id, buyer_ids in sim_sellers_matches.items()}

    # Empirical matches
    seller_empirical, seller_empirical_cubic = calculate_avg_std_df_with_cubic(
        matches_df, sellers_df, buyers_df, 'seller_ID', 'buyer_ID'
    )
    buyer_empirical, buyer_empirical_cubic = calculate_avg_std_df_with_cubic(
        matches_df, buyers_df, sellers_df, 'buyer_ID', 'seller_ID'
    )

    # Theoretical matches (convert dict to DataFrame)
    theoretical_matches_sellers = pd.DataFrame([
        {'seller_ID': k, 'buyer_ID': v} for k, buyers in sim_sellers_matches.items() for v in buyers
    ])
    theoretical_matches_buyers = pd.DataFrame([
        {'buyer_ID': k, 'seller_ID': v} for k, sellers in sim_buyers_matches.items() for v in sellers
    ])

    seller_theoretical, seller_theoretical_cubic = calculate_avg_std_df_with_cubic(
        theoretical_matches_sellers, sellers_df, buyers_df, 'seller_ID', 'buyer_ID'
    )
    buyer_theoretical, buyer_theoretical_cubic = calculate_avg_std_df_with_cubic(
        theoretical_matches_buyers, buyers_df, sellers_df, 'buyer_ID', 'seller_ID'
    )

    # Plot for sellers
    seller_plot = plot_comparison_with_cubic(
        seller_empirical, seller_theoretical, seller_empirical_cubic, seller_theoretical_cubic,
        x_label="Lawmaker's average valuation", y_label="Matched clients' average valuation",
        title="Lawmakers: distribution of matched agents",
        return_fig=True  # Ensure the plot is returned as a figure
    )

    # Plot for buyers
    buyer_plot = plot_comparison_with_cubic(
        buyer_empirical, buyer_theoretical, buyer_empirical_cubic, buyer_theoretical_cubic,
        x_label="Client's average valuation", y_label="Matched lawmakers' average valuation",
        title="Clients: distribution of matched agents",
        return_fig=True  # Ensure the plot is returned as a figure
    )

    return seller_plot, buyer_plot

def plot_comparison_with_cubic(empirical, theoretical, empirical_cubic, theoretical_cubic, 
                               x_label, y_label, title, return_fig=False):
    """
    Plot comparison between empirical and theoretical distributions with cubic fits.

    Parameters:
    - empirical: DataFrame containing empirical data with columns ['AgentVal', 'AvgPartnerVal', 'StdPartnerVal'].
    - theoretical: DataFrame containing theoretical data with columns ['AgentVal', 'AvgPartnerVal', 'StdPartnerVal'].
    - empirical_cubic: Coefficients of cubic polynomial fit for empirical data (or None).
    - theoretical_cubic: Coefficients of cubic polynomial fit for theoretical data (or None).
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - return_fig: If True, return the matplotlib figure object instead of displaying the plot.

    Returns:
    - fig (optional): Matplotlib figure object if return_fig=True.
    """
    fig = plt.figure(figsize=(16, 8))

    # Empirical data
    if not empirical.empty:
        plt.errorbar(empirical['AgentVal'], empirical['AvgPartnerVal'], yerr=empirical['StdPartnerVal'],
                     fmt='o', label='Empirical match', color='blue', alpha=0.4)
        #plt.errorbar(empirical['AgentVal'], empirical['CountPartners'],# yerr=empirical['StdPartnerVal'],
        #             fmt='o', label='Empirical match', color='blue', alpha=0.4)
        

    # Theoretical data
    if not theoretical.empty:
        plt.errorbar(theoretical['AgentVal'], theoretical['AvgPartnerVal'], yerr=theoretical['StdPartnerVal'],
                     fmt='o', label='Theoretical match', color='orange', alpha=0.4)
        #plt.errorbar(theoretical['AgentVal'], theoretical['CountPartners'],# yerr=empirical['StdPartnerVal'],
        #             fmt='o', label='Theoretical match', color='orange', alpha=0.4)

    # Empirical cubic fit
    if empirical_cubic is not None:
        x = np.linspace(empirical['AgentVal'].min(), empirical['AgentVal'].max(), 200)
        y = np.polyval(empirical_cubic, x)
        plt.plot(x, y, color='darkblue', linestyle='--', label='Empirical valuations fit')

    # Theoretical cubic fit
    if theoretical_cubic is not None:
        x = np.linspace(theoretical['AgentVal'].min(), theoretical['AgentVal'].max(), 200)
        y = np.polyval(theoretical_cubic, x)
        plt.plot(x, y, color='darkorange', linestyle='--', label='Theoretical valuations fit')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if return_fig:
        return fig  # Return the figure object for further use (e.g., saving)
    else:
        plt.show()  # Display the plot directly
def plot_comparison(empirical, theoretical, x_label, y_label, title):
    """
    Plot the comparison of average and standard deviation between empirical and theoretical data.

    Parameters:
    - empirical: DataFrame with 'AgentVal', 'AvgPartnerVal', and 'StdPartnerVal' for empirical data.
    - theoretical: DataFrame with 'AgentVal', 'AvgPartnerVal', and 'StdPartnerVal' for theoretical data.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Plot title.

    Returns:
    - None (displays a plot)
    """
    plt.figure(figsize=(16, 8))

    # Empirical data
    if not empirical.empty:
        plt.errorbar(empirical['AgentVal'], empirical['AvgPartnerVal'], yerr=empirical['StdPartnerVal'],
                     fmt='o', label='Empirical', color='blue', alpha=0.7)

    # Theoretical data
    if not theoretical.empty:
        plt.errorbar(theoretical['AgentVal'], theoretical['AvgPartnerVal'], yerr=theoretical['StdPartnerVal'],
                     fmt='o', label='Theoretical', color='orange', alpha=0.7)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_multi_model_forest_plot(data_list, save_path=None, dpi=300, font_size=14):
    """
    Create a forest plot with multiple models/experiments, with group separators and group labels.
    
    Parameters:
    -----------
    data_list : list of pandas.DataFrame
        List of DataFrames, each representing a different model/experiment.
    save_path : str, optional
        File path to save the plot. If None, the plot is not saved.
    dpi : int, optional
        The resolution in dots per inch (DPI) for saving the plot.
    font_size : int, optional
        Font size for text elements in the plot.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for further customization or saving.
    """
    # Set global font size
    plt.rcParams.update({'font.size': font_size})
    
    # Combine and process data
    combined_data = []
    for i, df in enumerate(data_list):
        if 'Model' not in df.columns:
            df['Model'] = f'Model {i+1}'
        combined_data.append(df)
    
    data_combined = pd.concat(combined_data)
    data_sorted = data_combined.copy()
    data_sorted = data_sorted[::-1]  # Reverse ordering
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    groups = data_sorted['VariableGroup'].unique()
    models = data_sorted['Model'].unique()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    color_dict = dict(zip(models, colors))
    
    y_positions = []
    y_labels = []
    current_y = 0
    legend_handles = []
    legend_labels = []
    group_boundaries = []
    
    min_x=0
    for group in groups:
        group_data = data_sorted[data_sorted['VariableGroup'] == group]
        unique_variables = group_data['VariableName'].unique()
        
        group_start_y = current_y
        
        for variable in unique_variables:
            var_data = group_data[group_data['VariableName'] == variable]
            
            for model in models:
                model_var_data = var_data[var_data['Model'] == model]
                if not model_var_data.empty:
                    y_positions.append(current_y)
                    y_labels.append(var_data['VariableNameToDisplay'].iloc[0])
                    min_x = min(min_x, min(model_var_data['estimate']))
                    ax.errorbar(
                        x=model_var_data['estimate'].iloc[0],
                        y=current_y,
                        xerr=(
                            np.array([model_var_data['estimate'].iloc[0] - model_var_data['lower_ci'].iloc[0]]),
                            np.array([model_var_data['upper_ci'].iloc[0] - model_var_data['estimate'].iloc[0]])
                        ),
                        fmt='o',
                        color=color_dict[model],
                        capsize=5,
                        markersize=8
                    )
                    
                    if model not in legend_labels:
                        legend_handles.append(plt.Line2D([], [], color=color_dict[model], marker='o', linestyle=''))
                        legend_labels.append(model)
            
            current_y += 1  # Increment position for the next model
        
        group_boundaries.append((group, group_start_y, current_y - 1))  # Track group boundaries

    # Add horizontal dashed lines for group separation
    for i, (_, _, end_y) in enumerate(group_boundaries[:-1]):
        ax.axhline(y=end_y + 0.5, color='gray', linestyle='--', linewidth=1)
    
    # Add vertical group labels
    for group, start_y, end_y in group_boundaries:
        group_center_y = (start_y + end_y) / 2
        ax.text(
            x=min_x,  # Adjust the x position to place near the y-axis
            y=group_center_y,
            s=group,
            rotation=90,
            verticalalignment='center',
            horizontalalignment='center',
            fontweight='bold',
            fontsize=font_size + 2  # Slightly larger font for group labels
        )
    
    ax.axvline(x=0, color='red', linestyle='--')  # Reference line
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=font_size)  # Set font size for y-tick labels
    ax.legend(legend_handles, legend_labels, title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size)
    
    fig.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
# Example usage:
# compute_and_plot_mean_valuation(sim, matches_df, beta_vector_buyer, beta_vector_seller)


# Example usage:
# compute_and_plot_mean_valuation(sim, matches_df, beta_vector_buyer, beta_vector_seller)

# =============================================================================
# 
# def compare_avg_std_direct(matches_df, simulation):
#     """
#     Compare the average and standard deviation of matched partner valuations 
#     between empirical and theoretical data for buyers and sellers.
# 
#     Parameters:
#     - matches_df: DataFrame with 'buyer_ID' and 'seller_ID' for empirical matches.
#     - simulation: Simulation object with buyers_matches and sellers_matches attributes.
# 
#     Returns:
#     - None (displays plots for buyers and sellers)
#     """
#     # Extract buyer and seller data
#     buyers_data = simulation.buyers.data
#     sellers_data = simulation.sellers.data
# 
#     buyers_df = pd.DataFrame({
#         'ID': buyers_data['ID'],
#         'Valuation': buyers_data['Valuation']
#     })
#     sellers_df = pd.DataFrame({
#         'ID': sellers_data['ID'],
#         'Valuation': sellers_data['Valuation']
#     })
# 
#     # Empirical matches
#     empirical_matches_buyers = matches_df.groupby('buyer_ID')['seller_ID'].apply(set).to_dict()
#     empirical_matches_sellers = matches_df.groupby('seller_ID')['buyer_ID'].apply(set).to_dict()
# 
#     # Theoretical matches
#     theoretical_matches_buyers = {buyer_id: set(seller_ids) 
#                                    for buyer_id, seller_ids in simulation.buyers_matches.items()}
#     theoretical_matches_sellers = {seller_id: set(buyer_ids) 
#                                     for seller_id, buyer_ids in simulation.sellers_matches.items()}
# 
#     # Helper function to calculate average and standard deviation
#     def calculate_avg_std(agent_valuation_by_id, partner_valuation_by_id, match_dict):
#         data = []
#         for agent_id, partners in match_dict.items():
#             if partners:
#                 agent_val = agent_valuation_by_id[agent_id]
#                 partner_vals = [partner_valuation_by_id[p] for p in partners if p in partner_valuation_by_id]
#                 if partner_vals:
#                     avg = np.mean(partner_vals)
#                     std = np.std(partner_vals)
#                     data.append((agent_val, avg, std))
#         return pd.DataFrame(data, columns=['AgentVal', 'AvgPartnerVal', 'StdPartnerVal'])
# 
#     # Calculate for sellers
#     seller_empirical = calculate_avg_std(
#         seller_valuation_by_id := dict(zip(sellers_df['ID'], sellers_df['Valuation'])),
#         buyer_valuation_by_id := dict(zip(buyers_df['ID'], buyers_df['Valuation'])),
#         empirical_matches_sellers
#     )
#     seller_theoretical = calculate_avg_std(
#         seller_valuation_by_id,
#         buyer_valuation_by_id,
#         theoretical_matches_sellers
#     )
# 
#     # Calculate for buyers
#     buyer_empirical = calculate_avg_std(
#         buyer_valuation_by_id,
#         seller_valuation_by_id,
#         empirical_matches_buyers
#     )
#     buyer_theoretical = calculate_avg_std(
#         buyer_valuation_by_id,
#         seller_valuation_by_id,
#         theoretical_matches_buyers
#     )
# 
#     # Plot function
#     def plot_comparison(empirical, theoretical, x_label, y_label, title):
#         plt.figure(figsize=(16, 6))
# 
#         if not empirical.empty:
#             plt.errorbar(empirical['AgentVal'], empirical['AvgPartnerVal'], yerr=empirical['StdPartnerVal'],
#                          fmt='o', label='Empirical', color='blue', alpha=0.7)
# 
#         if not theoretical.empty:
#             plt.errorbar(theoretical['AgentVal'], theoretical['AvgPartnerVal'], yerr=theoretical['StdPartnerVal'],
#                          fmt='o', label='Theoretical', color='orange', alpha=0.7)
# 
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.title(title)
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.show()
# 
#     # Plot for sellers
#     plot_comparison(seller_empirical, seller_theoretical,
#                     x_label="Seller Valuation", y_label="Matched Buyers' Valuation",
#                     title="Sellers: Avg & Std of Matched Buyers' Valuations")
# 
#     # Plot for buyers
#     plot_comparison(buyer_empirical, buyer_theoretical,
#                     x_label="Buyer Valuation", y_label="Matched Sellers' Valuation",
#                     title="Buyers: Avg & Std of Matched Sellers' Valuations")
# 
# 
# def plot_distribution_with_intervals(matches_df, simulation, side='sellers', 
#                                      nbins=20, percentiles=[5,50,95]):
#     """
#     Plot median and percentile bands of matched partners' valuations as a function
#     of an agent's valuation. This creates a fan chart (median line + shaded 90% interval).
# 
#     Parameters:
#     - matches_df: DataFrame with 'buyer_ID' and 'seller_ID' for empirical matches.
#     - simulation: Simulation object with buyers_matches and sellers_matches attributes.
#     - side: 'sellers' or 'buyers', the perspective side.
#       If side='sellers', x-axis = Seller Valuation, y-axis = distribution of matched Buyers' valuations.
#       If side='buyers', x-axis = Buyer Valuation, y-axis = distribution of matched Sellers' valuations.
#     - nbins: Number of bins to create along the agent valuations.
#     - percentiles: Which percentiles to compute. Default: [5,50,95] for median + 90% interval.
# 
#     Returns:
#     - None (displays a plot)
#     """
# 
#     buyers_data = simulation.buyers.data
#     sellers_data = simulation.sellers.data
# 
#     buyers_df = pd.DataFrame({
#         'ID': buyers_data['ID'],
#         'Valuation': buyers_data['Valuation']
#     })
#     sellers_df = pd.DataFrame({
#         'ID': sellers_data['ID'],
#         'Valuation': sellers_data['Valuation']
#     })
# 
#     # Empirical matches
#     empirical_matches_buyers = matches_df.groupby('buyer_ID')['seller_ID'].apply(set).to_dict()
#     empirical_matches_sellers = matches_df.groupby('seller_ID')['buyer_ID'].apply(set).to_dict()
# 
#     # Theoretical matches
#     theoretical_matches_buyers = {buyer_id: set(seller_ids) 
#                                   for buyer_id, seller_ids in simulation.buyers_matches.items()}
#     theoretical_matches_sellers = {seller_id: set(buyer_ids) 
#                                    for seller_id, buyer_ids in simulation.sellers_matches.items()}
# 
#     buyer_valuation_by_id = dict(zip(buyers_df['ID'], buyers_df['Valuation']))
#     seller_valuation_by_id = dict(zip(sellers_df['ID'], sellers_df['Valuation']))
# 
#     if side == 'sellers':
#         # X-axis: Seller valuation
#         # Y-axis: distribution of matched Buyers' valuations
#         agent_valuation_by_id = seller_valuation_by_id
#         empirical_match_dict = empirical_matches_sellers
#         theoretical_match_dict = theoretical_matches_sellers
#         partner_valuation_by_id = buyer_valuation_by_id
#         x_label = "Seller Valuation"
#         y_label = "Matched Buyers' Valuation"
#     else:
#         # side == 'buyers'
#         # X-axis: Buyer valuation
#         # Y-axis: distribution of matched Sellers' valuations
#         agent_valuation_by_id = buyer_valuation_by_id
#         empirical_match_dict = empirical_matches_buyers
#         theoretical_match_dict = theoretical_matches_buyers
#         partner_valuation_by_id = seller_valuation_by_id
#         x_label = "Buyer Valuation"
#         y_label = "Matched Sellers' Valuation"
# 
#     def summarize_by_bins(match_dict):
#         # Create a DataFrame of agent valuation and all matched partner valuations
#         data = []
#         for agent_id, partners in match_dict.items():
#             if partners:
#                 agent_val = agent_valuation_by_id[agent_id]
#                 partner_vals = [partner_valuation_by_id[p] for p in partners if p in partner_valuation_by_id]
#                 if partner_vals:
#                     for val in partner_vals:
#                         data.append((agent_val, val))
#         df = pd.DataFrame(data, columns=['AgentVal', 'PartnerVal'])
#         if len(df) == 0:
#             return None
#         # Bin the agent valuations
#         df['bin'] = pd.qcut(df['AgentVal'], nbins, duplicates='drop')
#         # For each bin, compute percentiles of PartnerVal
#         summary = df.groupby('bin')['PartnerVal'].apply(
#             lambda x: np.percentile(x, percentiles)
#         )
#         # Extract bin edges
#         bin_edges = [interval.left for interval in summary.index.categories] + [summary.index.categories[-1].right]
#         bin_centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))
# 
#         # summary is a Series of arrays of percentiles
#         # Convert to DataFrame: rows = bins, columns = percentiles
#         summary_df = pd.DataFrame(summary.tolist(), index=bin_centers, columns=[f'p{p}' for p in percentiles])
#         summary_df.index.name = 'bin_center'
#         summary_df.reset_index(inplace=True)
#         return summary_df
# 
#     emp_summary = summarize_by_bins(empirical_match_dict)
#     theo_summary = summarize_by_bins(theoretical_match_dict)
# 
#     # Now plot
#     # We'll assume percentiles=[5,50,95], giving us a median line and a 90% band.
#     # If you changed percentiles, adjust accordingly.
#     median_col = 'p50' if 'p50' in emp_summary.columns else emp_summary.columns[len(emp_summary.columns)//2]
#     lower_col = 'p5' if 'p5' in emp_summary.columns else emp_summary.columns[0]
#     upper_col = 'p95' if 'p95' in emp_summary.columns else emp_summary.columns[-1]
# 
#     plt.figure(figsize=(8,6))
# 
#     # Empirical line and shading
#     if emp_summary is not None:
#         plt.plot(emp_summary['bin_center'], emp_summary[median_col], color='blue', label='Empirical Median')
#         plt.fill_between(emp_summary['bin_center'], emp_summary[lower_col], emp_summary[upper_col],
#                          color='blue', alpha=0.2, label='Empirical 90% Interval')
# 
#     # Theoretical line and shading
#     if theo_summary is not None:
#         plt.plot(theo_summary['bin_center'], theo_summary[median_col], color='orange', label='Theoretical Median')
#         plt.fill_between(theo_summary['bin_center'], theo_summary[lower_col], theo_summary[upper_col],
#                          color='orange', alpha=0.2, label='Theoretical 90% Interval')
# 
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(f"{side.capitalize()} Perspective: Distribution with Median and 90% Intervals")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     
# 
# def plot_cubic_binscatter(matches_df, simulation, side='sellers', nbins=20):
#     """
#     Create a binscatter plot and fit a cubic line through the binned averages.
# 
#     Parameters:
#     - matches_df: DataFrame with 'buyer_ID' and 'seller_ID' for empirical matches.
#     - simulation: Simulation object with buyers_matches and sellers_matches attributes.
#     - side: 'sellers' or 'buyers'.
#       If side='sellers', x-axis = Seller Valuation, y-axis = avg matched Buyers' valuation.
#       If side='buyers', x-axis = Buyer Valuation, y-axis = avg matched Sellers' valuation.
#     - nbins: Number of bins for the binscatter.
# 
#     Returns:
#     - None (displays a plot)
#     """
# 
#     buyers_data = simulation.buyers.data
#     sellers_data = simulation.sellers.data
# 
#     buyers_df = pd.DataFrame({
#         'ID': buyers_data['ID'],
#         'Valuation': buyers_data['Valuation']
#     })
#     sellers_df = pd.DataFrame({
#         'ID': sellers_data['ID'],
#         'Valuation': sellers_data['Valuation']
#     })
# 
#     # Empirical matches
#     empirical_matches_buyers = matches_df.groupby('buyer_ID')['seller_ID'].apply(set).to_dict()
#     empirical_matches_sellers = matches_df.groupby('seller_ID')['buyer_ID'].apply(set).to_dict()
# 
#     buyer_valuation_by_id = dict(zip(buyers_df['ID'], buyers_df['Valuation']))
#     seller_valuation_by_id = dict(zip(sellers_df['ID'], sellers_df['Valuation']))
# 
#     if side == 'sellers':
#         # X-axis: Seller valuation, Y-axis: avg matched buyers' valuation
#         agent_valuation_by_id = seller_valuation_by_id
#         match_dict = empirical_matches_sellers
#         partner_valuation_by_id = buyer_valuation_by_id
#         x_label = "Seller Valuation"
#         y_label = "Average Matched Buyers' Valuation"
#     else:
#         # side == 'buyers'
#         # X-axis: Buyer valuation, Y-axis: avg matched sellers' valuation
#         agent_valuation_by_id = buyer_valuation_by_id
#         match_dict = empirical_matches_buyers
#         partner_valuation_by_id = seller_valuation_by_id
#         x_label = "Buyer Valuation"
#         y_label = "Average Matched Sellers' Valuation"
# 
#     # Create data for binscatter
#     data = []
#     for agent_id, partners in match_dict.items():
#         if partners:
#             agent_val = agent_valuation_by_id[agent_id]
#             partner_vals = [partner_valuation_by_id[p] for p in partners if p in partner_valuation_by_id]
#             if partner_vals:
#                 # Compute the average partner valuation for this agent
#                 avg_partner_val = np.mean(partner_vals)
#                 data.append((agent_val, avg_partner_val))
# 
#     df = pd.DataFrame(data, columns=['AgentVal', 'PartnerVal'])
#     if len(df) == 0:
#         print("No data to plot.")
#         return
# 
#     # Bin the agent valuations
#     df['bin'] = pd.qcut(df['AgentVal'], nbins, duplicates='drop')
# 
#     # Compute mean PartnerVal in each bin
#     bin_summary = df.groupby('bin')['PartnerVal'].mean()
#     # Extract bin edges and compute bin centers
#     bin_edges = [interval.left for interval in bin_summary.index.categories] + [bin_summary.index.categories[-1].right]
#     bin_centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))
# 
#     mean_vals = bin_summary.values
# 
#     # Plot the binscatter (points at bin centers)
#     plt.figure(figsize=(8,6))
#     plt.scatter(bin_centers, mean_vals, color='blue', alpha=0.7, label='Binned Averages')
# 
#     # Fit a cubic polynomial to the binned averages
#     # degree=3 for cubic
#     coeffs = np.polyfit(bin_centers, mean_vals, deg=3)
#     poly = np.poly1d(coeffs)
# 
#     # Create a smooth curve for the cubic fit
#     x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
#     y_smooth = poly(x_smooth)
# 
#     # Plot the cubic line
#     plt.plot(x_smooth, y_smooth, color='red', label='Cubic Fit')
# 
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(f"{side.capitalize()} Perspective: Cubic Binscatter")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# =============================================================================
