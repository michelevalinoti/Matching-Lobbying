#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:08:25 2024

@author: michelev
"""

# main.py
import numpy as np
import pandas as pd
import os
import sys
import logging
import json
from scipy.stats import uniform, norm, truncnorm
#mport matplotlib.pyplot as plt
from market_simulation import MarketSide, MarketSimulation
from moment_estimator import MomentEstimator, compute_moments_from_simulation, compute_moments_from_data, simulate_and_compute # Import the MomentEstimator class
from Base import get_truncnorm_params, normalize_columns, optimize_result_to_serializable_dict, save_optimize_result_to_json
import multiprocessing
from plots import compute_and_plot_mean_valuation, create_multi_model_forest_plot

def parse_job_idx(job_idx, bootstrap = False):
    """
    job_idx: integer in [1..400]
    Returns:
        scenario: integer in [1..4]
        b: bootstrap replicate index in [1..100]
    """
    if bootstrap==False:
        return job_idx, None
    scenario = (job_idx - 1) // 100 + 1  # block of 100 => scenario
    b = (job_idx - 1) % 100 + 1         # remainder => 1..100
    return scenario, b

#%%

machine = '/Users/michelev/Dropbox/EmpiricalMatchingPlatform/'
#machine = '/home/mv2164/EmpiricalMatchingPlatform/'
imported_tables_folder = os.path.join(machine, 'tables')
# Add the directory containing 'market_simulation.py' to sys.path
#script_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(script_dir)

#JOB_IDX = int(os.environ["SLURM_ARRAY_TASK_ID"])

#%%
 

def example():
    
#     alpha = -0.3
#     N = 100  # Number of participants
#     coefficient_A = 0
#     coefficient_B = 0
#     data = np.random.normal(0, 1, N)
    
#     v_upper_A = 0.4  # max(np.max(buyers.get_valuations()), np.max(sellers.get_valuations()))
#     v_upper_B = 0.3
#     v_lower_A = 0.1
#     v_lower_B = 0.0

#     distribution_A = uniform(v_lower_A, -v_lower_A+v_upper_A)  # norm(loc=0, scale=1)
#     distribution_B = uniform(v_lower_B, -v_lower_B+v_upper_B) 
    
#     buyers = MarketSide(name='Buyers', N=N, distribution=distribution_A, v_upper=v_upper_A, v_lower=v_lower_A)
#     sellers = MarketSide(name='Sellers', N=N, distribution=distribution_B, v_upper=v_upper_B, v_lower=v_lower_B)

#     buyers.update_valuations(data, coefficient_A)
#     sellers.update_valuations(data, coefficient_B)

#     simulation = MarketSimulation(buyers=buyers, sellers=sellers, h='P')

#     simulation.check_chi_derivatives()
#     simulation.evaluate_Delta()
#     simulation.compute_omegas()
#     print("Buyers' omega:", simulation.buyers_omega)
#     print("Sellers' omega:", simulation.sellers_omega)

#     print("Buyers' threshold omega:", simulation.buyers_threshold_omega)
#     print("Sellers' threshold omega:", simulation.sellers_threshold_omega)

#     simulation.compute_thresholds()
#     #print("Buyers' thresholds:", simulation.buyers_thresholds)
#     #print("Sellers' thresholds:", simulation.sellers_thresholds)

#     simulation.compute_matches()
#     simulation.plot_matches(save_path=machine + 'figures/08232024_matches_A_B.pdf')
#     # print("Buyers' matches:", simulation.buyers_matches)
#     # print("Sellers' matches:", simulation.sellers_matches)

#     simulation.compute_payments()
#     simulation.plot_payments(save_path=machine + 'figures/08232024_payments_A_B.pdf')
#     #print("Buyers' payments:", simulation.buyers_payments)

#     #simulation.compute_theoretical_values()

#     # Plotting the results
#     #simulation.plot_buyers_payments()
#     #simulation.plot_sellers_payments()
#     #simulation.plot_buyers_threshold_function()
#     #simulation.plot_sellers_threshold_function()
#     #simulation.plot_matches(


#     N_buyers = 100
#     N_sellers = 100
    
#     K_buyers = 1
#     K_sellers = 1
    
#     ID_buyers = np.arange(N_buyers)
#     ID_sellers = np.arange(N_sellers)
    
#     beta_true_buyers = np.array([2])
#     beta_true_sellers = np.array([1])
    
    
#     alpha = -0.5

#     sigma = 1
    
#     # Create the distributions using get_truncnorm_params
    
#     data_distribution_buyers = truncnorm(*get_truncnorm_params(0, -2, 2, 1))
#     data_distribution_sellers = truncnorm(*get_truncnorm_params(0, -2, 2, 1))
    
#     #data_distribution_buyers = truncnorm(*get_truncnorm_params(3+.15, 1+.3, 5+.3, 1))
#     #data_distribution_sellers = truncnorm(*get_truncnorm_params(2+.15, 0+.3, 4+.3, 1))

#     # For normal distributions, just use the mean and scale
    
#     means_buyers = np.array([3])
#     means_sellers = np.array([4])
     
#     #error_distribution_buyers = truncnorm(*get_truncnorm_params(0, 0, 1e-3, 1))
#     #error_distribution_sellers = truncnorm(*get_truncnorm_params(0, 0, 1e-3, 1))

#     error_distribution_buyers = truncnorm(*get_truncnorm_params(0, -1, 1, 1))
#     error_distribution_sellers = truncnorm(*get_truncnorm_params(0, -1, 1, 1))

#     # Define a more extensive beta grid centered around the true values
#     beta_values_buyers = np.linspace(1, 3, 5)
#     beta_values_sellers = np.linspace(0.5, 1.5, 5)

#     ID_market_buyers = np.arange(N_buyers)
#     ID_market_sellers = np.arange(N_sellers)
    
#     # Instantiate the MomentEstimator with scipy.stats distributions
#     estimator = MomentEstimator(
#         N_buyers = N_buyers,
#         N_sellers = N_sellers,
        
#         K_buyers = K_buyers,
#         K_sellers = K_sellers,
        
#         ID_buyers = ID_market_buyers,
#         ID_sellers = ID_market_sellers,
        
#         sigma = sigma,
        
#         error_generator_buyers = error_distribution_buyers,
#         error_generator_sellers = error_distribution_sellers,
        
#         X_buyers = None,
#         X_sellers = None,
        
#         beta_true_buyers = beta_true_buyers,
#         beta_true_sellers = beta_true_sellers,
#         data_generator_buyers = data_distribution_buyers,
#         data_generator_sellers = data_distribution_sellers, 
#         means_buyers = means_buyers,
#         means_sellers = means_sellers
#     )
 
#     eta = 25
#     # Generate the true data and get the buyers and sellers
#     buyers, sellers = estimator.generate_data_and_sides(eta)
       
   
#     #simulation = MarketSimulation(buyers=buyers, sellers=sellers, h='P')
#     #simulation.check_chi_derivatives()
#     #simulation.evaluate_Delta()
#     #simulation.compute_omegas()
#     # # Compute the true moments and simulate the true equilibrium
#     simulation = estimator.simulate_empirical_equilibrium(buyers, sellers)
#     # Plot the matches for the true equilibrium
#     simulation.plot_matches(thresholds="empirical")#, save_path=machine + 'figures/matches_true_equilibrium_08232024.pdf')

#     true_moments = estimator.compute_moments_from_simulation(simulation)
#     # Simulate the equilibrium for several beta values and compute the moments
#     # Simulate the equilibrium for several beta values and compute the average moments
#     S = 10  # Number of simulations to average
    
#     moments_folder = os.path.join(machine, 'moments/')
#     # Simulate the equilibrium for several beta values and compute the average moments
#     for beta_b in beta_values_buyers:
#         for beta_s in beta_values_sellers:
#             print(f"Running simulations for beta_b={beta_b}, beta_s={beta_s}")
            
#             # Create a unique filename for this beta_b and beta_s pair
#             output_file = os.path.join(moments_folder, f'08222024_moments_diff_beta_b_{beta_b}_beta_s_{beta_s}.csv')
            
#             # Check if this file already exists to avoid recomputation
#             if not os.path.exists(output_file):
#                 # Compute average simulated moments for S simulations
#                 average_simulated_moments = estimator.simulate_average_moments_parallelized(beta_b, beta_s, eta, S)
                
#                 # Compute the difference between true and simulated moments
#                 moment_diff = estimator.compute_moment_differences(true_moments, average_simulated_moments)
                
#                 # Compute the distance (Euclidean norm of differences)
#                 moment_diff['distance'] = np.sqrt(moment_diff["moment_buyers_diff"]**2 + moment_diff["moment_sellers_diff"]**2)
                
#                 # Prepare data to save
#                 data_to_save = {
#                     'beta_b': [beta_b],
#                     'beta_s': [beta_s],
#                     'moment_buyers_diff': [moment_diff['moment_buyers_diff']],
#                     'moment_sellers_diff': [moment_diff['moment_sellers_diff']],
#                     'distance': [moment_diff['distance']]
#                 }
                
#                 # Save the data to a CSV file
#                 pd.DataFrame(data_to_save).to_csv(output_file, index=False)
#                 print(f"Results saved to {output_file}")

#     distances = pd.DataFrame(index=beta_values_buyers, columns=beta_values_sellers)
#     directory = machine + 'moments/'  # Directory where the CSV files are saved

#     # Loop through all files in the directory
#     for filename in os.listdir(directory):
#         if filename.startswith("08222024_moments"):
#             # Extract beta_b and beta_s from the filename
#             parts = filename.split('_')
#             beta_b = float(parts[5])
#             beta_s = float(parts[8].replace('.csv', ''))

#             # Read the CSV file
#             df = pd.read_csv(os.path.join(directory, filename))

#             # Extract the distance and store it in the DataFrame
#             distance = df['distance'].values[0]
#             distances.at[beta_b, beta_s] = distance

#     # Convert the DataFrame to a 2D numpy array
#     distances_array = distances.values.astype(float)

#     # Plot the heatmap
#     plt.figure(figsize=(10, 8))
#     plt.contourf(beta_values_buyers, beta_values_sellers, distances_array, cmap='viridis')
#     plt.colorbar(label=r'$||\psi^\beta(x) - \psi(x)||$')
#     plt.xlabel(r'$\beta_B$')
#     plt.ylabel(r'$\beta_A$')
#     #plt.title('Distance between Moments for Different Beta Values')
#     #plt.savefig(output_file)
#     #plt.show()
#     plt.savefig(directory + '08232024_moment_distances.pdf', format='pdf', bbox_inches='tight')

# # Set the parameters
#     machine = '/Users/michelev/Dropbox/EmpiricalMatchingPlatform/'
#     directory = machine + 'moments/'  # Directory where the CSV files are saved
#     output_file = machine + 'figures/moments_distance_heatmap.png'
#     beta_values_buyers = np.linspace(0.5, 1.5, 5)  # Replace with your actual beta values for buyers
#     beta_values_sellers = np.linspace(1.5, 2.5, 5)  # Replace with your actual beta values for sellers

# # Generate the heatmap
# #generate_heatmap_from_csv(directory, output_file, beta_values_buyers, beta_values_sellers)
#     # # Convert the list to a DataFrame
#     # distance_data_df = pd.DataFrame(moments_distances, columns=['beta_b', 'beta_s', 'distance'])
    
#     # # Pivot the DataFrame to get the correct shape for plotting
#     # distance_matrix = distance_data_df.pivot(index='beta_s', columns='beta_b', values='distance')
    
#     # # Plot the distance heatmap
#     # plt.figure(figsize=(10, 8))
#     # plt.contourf(distance_matrix.columns, distance_matrix.index, distance_matrix.values, cmap='viridis')
#     # plt.colorbar(label='Distance between moments')
#     # plt.xlabel('Beta Buyers')
#     # plt.ylabel('Beta Sellers')
#     # plt.title('Distance between Moments for Different Beta Values')
#     # plt.savefig(machine + 'figures/moments_distance.png')
#     # plt.show()
    
    
#     N_buyers = 123 # number of clients x market in the matching dataset
#     N_sellers = 203 # number of politicians x market in the matching dataset
    
#     K_buyers = 8 # number of variables of clients in the matching dataset
#     K_sellers = 10 # number of variables of politicians in the matching dataset 
    
#     ID_buyers = np.arange(K_buyers) # IDs of clients
#     ID_sellers = np.arange(K_sellers) # IDs of politicians
    
#     estimator = MomentEstimator(
#         N_buyers = N_buyers,
#         N_sellers = N_sellers,
        
#         K_buyers = K_buyers,
#         K_sellers = K_sellers,
        
#         ID_buyers = ID_buyers,
#         ID_sellers = ID_sellers,
        
#         sigma = sigma,
        
#         error_generator_buyers = error_distribution_buyers,
#         error_generator_sellers = error_distribution_sellers,
        
#         X_buyers = None,
#         X_sellers = None,
        
#         beta_true_buyers = None,
#         beta_true_sellers = None,
#         data_generator_buyers = None,
#         data_generator_sellers = None, 
#         means_buyers = None,
#         means_sellers = None
#     )
 
#     eta = 50
       
#     buyers, sellers = estimator.generate_sides()
    
    
    
    N_clients = 200
    N_politicians = 200
    
    K_clients = 1
    K_politicians = 1
    
    
    beta_true_clients = np.array([1])
    beta_true_politicians = np.array([2])
    
    h = 'P'
    #alpha = -0.5
    

    # Create the distributions using get_truncnorm_params
    
    data_distribution_clients = truncnorm(*get_truncnorm_params(0, -2, 2, 1))
    data_distribution_politicians = truncnorm(*get_truncnorm_params(0, -2, 2, 1))
    
    means_clients = np.array([3])
    means_politicians = np.array([4])
     
    
    error_distribution_clients= truncnorm(*get_truncnorm_params(0, -2, 2, 1))
    error_distribution_politicians = truncnorm(*get_truncnorm_params(0, -2, 2, 1))

    beta_values_clients = np.linspace(1, 3, 5)
    beta_values_politicians = np.linspace(0.5, 1.5, 5)

    eta = 50
    
    simulated_data_params = {
        'N_buyers': N_clients,  # Number of buyers
        'N_sellers': N_politicians,  # Number of sellers
        'K_buyers': K_clients,     # Number of predictors for buyers
        'K_sellers': K_politicians,    # Number of predictors for sellers
        'beta_true_buyers': beta_true_clients,  # True beta coefficients for buyers
        'beta_true_sellers': beta_true_politicians,  # True beta coefficients for sellers
        'means_buyers': means_clients,  # Mean of predictors for buyers
        'means_sellers': means_politicians  # Mean of predictors for sellers
    }
    
    estimator = MomentEstimator(
        
        observed_data_buyers=None,
        observed_data_sellers=None,
        error_generator_buyers = error_distribution_clients,
        error_generator_sellers = error_distribution_politicians,
        simulated_data_params=simulated_data_params,
        eta=eta
    )
 
    simulation = estimator.simulate_empirical_equilibrium(h)
    
    theoretical_moments = compute_moments_from_simulation(simulation['single_market'].buyers.data, simulation['single_market'].sellers.data, simulation['single_market'])
    
    S = 2 # Number of simulations for averaging moments in the objective function
    method = 'Nelder-Mead'
    options = {'maxiter': 20, 'adaptive':True, 'disp': True}
    bounds = None
    # Initial guess for beta values (you can adjust these)
    # Assuming beta_initial is a numpy array of shape (2,)
    beta_initial = np.array([0,0])  # Shape: (2,)
    
    
    #W = np.eye(2*2*K_clients * K_politicians + 2*K_clients + 2*K_politicians) # Weight matrix, identity for simplicity
   
    
    # Run the estimation
    #result = estimator.estimate_betas(beta_initial, theoretical_moments, K_clients, K_politicians, eta, h, S, W, method, bounds, options)
    
    # Extract the estimated beta values
    #beta_estimated = result.x
    #beta_buyers_estimated, beta_sellers_estimated = beta_estimated
    
    
    #print(beta_buyers_estimated, flush=True)
    #print(beta_sellers_estimated, flush=True)
    
    
        
        
        # Define the grid ranges
    beta_buyers_range = np.linspace(0.5, 1.5, 5)       # Around true beta_clients = 1
    beta_sellers_range = np.linspace(1.5, 2.5, 5)      # Around true beta_politicians = 2
    
    # Create a meshgrid
    beta_buyers_grid, beta_sellers_grid = np.meshgrid(beta_buyers_range, beta_sellers_range)
    
    # Flatten the grids for iteration
    beta_buyers_flat = beta_buyers_grid.flatten()
    beta_sellers_flat = beta_sellers_grid.flatten()
    
    # Initialize an array to store objective function values
    objective_values = {}#np.zeros_like(beta_buyers_flat)
    
    W = np.eye(4)#2*2*K_clients * K_politicians + 2*K_clients + 2*K_politicians) # Weight matrix, identity for simplicity
    # Weight matrix, identity for simplicity
    S = 5 # Number of simulations for averaging moments; keep low for speed
    
    # Start the grid evaluation
    for idx, (beta_b, beta_s) in enumerate(zip(beta_buyers_flat, beta_sellers_flat)):
        # Combine beta_buyers and beta_sellers into a single array
        beta = np.array([beta_b, beta_s])
        
        # Compute the objective function
        obj_value = estimator.objective_function(
            beta=beta, 
            true_moments=theoretical_moments, 
            K_buyers=K_clients, 
            K_sellers=K_politicians, 
            eta=eta, 
            h=h, 
            S=S,
            W=W
        )
        
        # Store the objective value
        objective_values[(beta_b, beta_s)] = obj_value
        
        print(f"Evaluated objective at beta_buyers={beta_b:.2f}, beta_sellers={beta_s:.2f}: {obj_value:.4f}", flush=True)
        
         
    # Convert the dictionary into arrays for plotting
    beta_buyers = []
    beta_sellers = []
    objective_values_list = []
    
    for (beta_b, beta_s), value in objective_values.items():
        beta_buyers.append(beta_b)
        beta_sellers.append(beta_s)
        objective_values_list.append(np.log(value))  # Use log of the values for better scaling
    
    # Convert to numpy arrays
    beta_buyers = np.array(beta_buyers)
    beta_sellers = np.array(beta_sellers)
    objective_values_array = np.array(objective_values_list)
        
       
    # Perform KDE
    values = np.vstack([beta_buyers, beta_sellers])
    weights = objective_values_array
    kde = gaussian_kde(values, weights=weights)
    
    # Create a grid for contour plotting
    x_min, x_max = beta_buyers.min(), beta_buyers.max()
    y_min, y_max = beta_sellers.min(), beta_sellers.max()
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    # Use serif font for an academic look
    plt.rcParams['font.family'] = 'serif'         # Use Times or LaTeX-like fonts
    plt.rcParams['font.size'] = 12               # Set default font size
    plt.rcParams['axes.titlesize'] = 14          # Title font size
    plt.rcParams['axes.labelsize'] = 12          # Axes Labels font size
    plt.rcParams['legend.fontsize'] = 10         # Legend font size
    contour = plt.contourf(X, Y, Z, levels=10, cmap='viridis')  # Filled contour
    plt.colorbar(contour, label='Moment distance')  # Add colorbar
    
    # Mark the true parameters
    beta_true_clients = 1.0
    beta_true_politicians = 2.0
    plt.scatter(beta_true_clients, beta_true_politicians, color='red', marker='x', s=100, label='True Parameters')
    
    # Add labels, title, and legend
    plt.xlabel(r'$\beta_\text{buyers}$')  # Updated x-axis label
    plt.ylabel(r'$\beta_\text{sellers}$') # Updated y-axis label

    #plt.title('Objective Function KDE Contour')
    plt.legend()
    
    plt.savefig(os.path.join(machine,'moments_distances.png'), dpi=300)
# Run the main function

def empirical_estimation():
    
    # ONLY VENDORS AND SOME ATTRIBUTES

    # Load the buyers, sellers, and matches datasets
    print("Running main() ...", flush = True)
    
    global_estimation = True
    local_estimation = False
    bootstrapping = False
    
    JOB_IDX = 4#int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    
    print(f"JOB_IDX = {JOB_IDX}", flush=True)
    
    #JOB_IDX = 4
    all_attributes = None
    vendors_only = None
    
    SCENARIO, BOOT_IDX = parse_job_idx(JOB_IDX, bootstrapping)
    
    print(f"SCENARIO: {SCENARIO}", flush = True)
    print(f"BOOT_IDX: {BOOT_IDX}...", flush = True)
    
    if SCENARIO == 1:
        all_attributes = True
        vendors_only = True
    elif SCENARIO == 2:
        all_attributes = False
        vendors_only = True
    elif SCENARIO == 3:
        all_attributes = True
        vendors_only = False
    elif SCENARIO == 4:
        all_attributes = False
        vendors_only = False
        
    print(f"All attributes: {all_attributes}", flush=True)

    print(f"Vendors only: {vendors_only}", flush=True)

    # Could parse arguments or set up logging, etc.
    
    if vendors_only:
        # Import datasets for matching
        clients_df = pd.read_csv(os.path.join(imported_tables_folder, "buyers_vendors_table.csv"))
        legislators_df = pd.read_csv(os.path.join(imported_tables_folder, "sellers_table_imputed.csv"))
        matches_df = pd.read_csv(os.path.join(imported_tables_folder, "matches_vendors_table.csv"))
        
        # Define the columns to normalize explicitly for each dataframe
        clients_columns_to_normalize = ['asset_amount',
                                        'income_amount',
                                        'log_asset_amount',
                                        'log_income_amount']
    
        clients_df = normalize_columns(clients_df, clients_columns_to_normalize)

    else:
        clients_df = pd.read_csv(os.path.join(imported_tables_folder, "buyers_table.csv"))
        legislators_df = pd.read_csv(os.path.join(imported_tables_folder, "sellers_table_imputed.csv"))
        matches_df = pd.read_csv(os.path.join(imported_tables_folder, "matches_table.csv"))
        
    matches_df['Market'] = matches_df['year'].copy().astype(str)
    matches_df = matches_df.loc[matches_df.flag_contacts_intermediated,:]
    
    if vendors_only:
            
        selected_buyers_predictors = [
            'is_retained',
            'log_asset_amount_norm',
        ]
        
    else:
        
        selected_buyers_predictors = [
            'is_retained'
        ]
        
    
        
    if all_attributes:
        # List of possible predictors for sellers
        # Comment out the predictors you do not want to include
        selected_seller_predictors = [
            'connection_individual_lobbyist',  # Binary: 1 if connected to individual lobbyist, else 0
            'revolving_flag',                  # Binary: 1 if revolving door, else 0
            'is_chair',                        # Binary: 1 if chair, else 0
            'appropriations_committee',        # Binary: 1 if on appropriations committee, else 0
            #'distinct_committee_count',        # Numeric: Number of distinct committees
            #'terms',                           # Numeric: Number of terms served
            'senator',                         # Binary: 1 if senator, else 0
            #'freshman',                        # Binary: 1 if freshman, else 0
            'minority_party',                  # Binary: 1 if minority party, else 0
            #'incumbent_status',                # Binary: 1 if incumbent, else 0
            #'cfscore_rec',                     # Numeric: Campaign finance score (recurring)
            'cfscore_con',                     # Numeric: Campaign finance score (contribution)
            #'BUSNS',                           # Numeric: Business-related score
            'GOVT',                            # Numeric: Government-related score
            #'LES',                             # Numeric: Legislative effectiveness score
            'share_votes_general'              # Numeric: Share of votes in general elections
        ]
        
    else:
        # List of possible predictors for sellers
        # Comment out the predictors you do not want to include
        selected_seller_predictors = [
            'connection_individual_lobbyist',  # Binary: 1 if connected to individual lobbyist, else 0
            'revolving_flag',                  # Binary: 1 if revolving door, else 0
            #'is_chair',                        # Binary: 1 if chair, else 0
            #'appropriations_committee',        # Binary: 1 if on appropriations committee, else 0
            #'distinct_committee_count',        # Numeric: Number of distinct committees
            #'terms',                           # Numeric: Number of terms served
            #'senator',                         # Binary: 1 if senator, else 0
            #'freshman',                        # Binary: 1 if freshman, else 0
            #'minority_party',                  # Binary: 1 if minority party, else 0
            #'incumbent_status',                # Binary: 1 if incumbent, else 0
            #'cfscore_rec',                     # Numeric: Campaign finance score (recurring)
            #'cfscore_con',                     # Numeric: Campaign finance score (contribution)
            #'BUSNS',                           # Numeric: Business-related score
            #'GOVT',                            # Numeric: Government-related score
            'LES',                             # Numeric: Legislative effectiveness score
            #'share_votes_general'              # Numeric: Share of votes in general elections
        ]
        
    year_cols = []
    
    
    for year in [2019, 2020, 2021, 2022]:
        
        clients_df['year_'+str(year)] = clients_df['year'] == year
        legislators_df['year_'+str(year)] = legislators_df['year'] == year
        year_cols = year_cols + ['year_'+str(year)]
        
    
    selected_buyers_predictors = selected_buyers_predictors + year_cols
    selected_seller_predictors = selected_seller_predictors + year_cols
    
    # Define the number of predictors
    K_clients = len(selected_buyers_predictors)         # 'retained'
    K_politicians = len(selected_seller_predictors)       # 'connection_individual_lobbyist' and 'revolving_flag'
    
    print(f"Number of selected buyer predictors (K_buyers): {K_clients}", flush=True)
    print(f"Number of selected seller predictors (K_sellers): {K_politicians}", flush=True)
    
    def return_structured_data(clients_df, legislators_df):
        # Prepare buyers' predictors
        buyers_predictors = clients_df[selected_buyers_predictors].astype(float).values
        #buyers_predictors = np.hstack((np.ones((buyers_predictors.shape[0], 1)), buyers_predictors))
        sellers_predictors = legislators_df[selected_seller_predictors].astype(float).values
        #sellers_predictors = np.hstack((np.ones((sellers_predictors.shape[0], 1)), sellers_predictors))
        
        # Define structured data types
        buyer_dtype = np.dtype([
            ('ID', 'U50'),                # Using Unicode string for IDs like "5_Retained"
            ('Market', 'U50'),
            ('Predictors', np.float64, (K_clients,)),
            ('Valuation', np.float64)
        ])
        
        seller_dtype = np.dtype([
            ('ID', 'U100'),                # Using Unicode string for IDs like "5_PAC_Yes_revolving_Yes"
            ('Market', 'U50'),
            ('Predictors', np.float64, (K_politicians,)),
            ('Valuation', np.float64)
            
        ])
        
        # Create structured arrays
        buyers_structured = np.zeros(len(clients_df), dtype=buyer_dtype)
        sellers_structured = np.zeros(len(legislators_df), dtype=seller_dtype)
        
        # Assign values
        buyers_structured['ID'] = clients_df['buyer_ID']
        buyers_structured['Predictors'] = buyers_predictors
        buyers_structured['Market'] = clients_df['year']
        # Valuations will be generated later
        
        sellers_structured['ID'] = legislators_df['seller_ID']
        sellers_structured['Predictors'] = sellers_predictors
        sellers_structured['Market'] = legislators_df['year']
    
        return buyers_structured, sellers_structured
    
    buyers_structured, sellers_structured = return_structured_data(clients_df, legislators_df)
    error_distribution_clients= truncnorm(*get_truncnorm_params(1, -2, 2, 1))
    error_distribution_politicians = truncnorm(*get_truncnorm_params(1, -2, 2, 1))
    
    eta = 100
    print(f"Eta: {eta}")
    h='P'     
    S=5
    W = np.eye(2*(2*(K_clients-4) * (K_politicians-4) + K_clients-4 + K_politicians-4))

    estimator_empirical = MomentEstimator(
        
        observed_data_buyers=buyers_structured,
        observed_data_sellers=sellers_structured,
        error_generator_buyers = error_distribution_clients,
        error_generator_sellers = error_distribution_politicians,
        simulated_data_params=None,
        eta=eta
    )
    
    empirical_moments = compute_moments_from_data(matches_df, buyers_structured, sellers_structured)
    opt_global_filename = os.path.join(machine, f'estimation_results/intermediaries_global_estimation_vendors_only_{vendors_only}_all_attributes_{all_attributes}_eta_{eta}_S_{S}.json')
    opt_local_filename = os.path.join(machine, f'estimation_results/intermediaries_local_estimation_vendors_only_{vendors_only}_all_attributes_{all_attributes}_eta_{eta}_S_{S}.json')

    if global_estimation:
        
        if vendors_only:
            if all_attributes:
                beta_bounds = [#(0,10),
                               (-5,10),
                               (0,10),
                               
                               (-15,15),
                               (-15,15),
                               (-15,15),
                               (-15,15),
                               
                               #(0,10),
                               (-10,10),
                               (-10,10),
                               (0,10),
                               (0,10),
                               (0,10),
                               (-10,0),
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               
                               (-15,15),
                               (-15,15),
                               (-15,15),
                               (-15,15)
                               ]
            else:
                beta_bounds = [#(0,10),
                               (-5,10),
                               (0,10),
                               
                               (-15,15),
                               (-15,15),
                               (-15,15),
                               (-15,15),
                               
                               
                               #(0,10),
                               (-10,10),
                               (-10,10),
                               (0,10),
                               
                               (-15,15),
                               (-15,15),
                               (-15,15),
                               (-15,15)
                               ]
        
        else:
            if all_attributes:
                beta_bounds = [#(0,10),
                               (-5,10),
                               
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               
                               #(0,10),
                               (-10,10),
                               (-10,10),
                               (0,10),
                               (0,10),
                               (0,10),
                               (-10,0),
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               (-10,10)
                               ]
            else:
                beta_bounds = [#(0,10),
                               (-5,10),
                               
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               
                               #(0,10),
                               (-10,10),
                               (-10,10),
                               (0,10),
                               
                               (-10,10),
                               (-10,10),
                               (-10,10),
                               (-10,10)
                               ]
        
        workers = 1    
        seeds = None
        
        #logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_global_all_years.log')
        #estimator_empirical.set_logger(name='estimation_vendors_global_all_years', log_file_path=logger_path)
    
        global_opt = estimator_empirical.estimate_betas_global(beta_bounds, empirical_moments, K_clients, K_politicians, eta, h, S, W, seeds, workers)
        
        save_optimize_result_to_json(global_opt, opt_global_filename, exclude_attrs = ['hess', 'hess_inv'])
     
    if local_estimation:
        
        print("Local estimation", flush = True)
        
        method = 'Nelder-Mead'
        options = {'maxiter': 20,
                   #'maxfev': 1,
                   'adaptive':True,
                   'disp': True}
        
        if not bootstrapping:
            
            print("Initial estimation", flush = True)
            
            with open(opt_global_filename, 'r') as file:
                global_opt_result = json.load(file)
                
            beta_global = global_opt_result['x']
            
            eta = 100
            bounds = None
            seeds = None
            h='P'     
            S=5
            W = np.eye(2*(2*(K_clients-4) * (K_politicians-4) + K_clients-4 + K_politicians-4))

            local_opt = estimator_empirical.estimate_betas_local(beta_global, empirical_moments, K_clients, K_politicians, eta, h, S, W, seeds, method, bounds, options)
            save_optimize_result_to_json(local_opt, opt_local_filename, exclude_attrs = ['hess', 'hess_inv', 'final_simplex'])

        else:
            
            S=1
            for BOOT_IDX in range(1,101):
            
                opt_local_filename_boot = os.path.join(machine, f'estimation_results/intermediaries_local_estimation_boot_{BOOT_IDX}_vendors_only_{vendors_only}_all_attributes_{all_attributes}_eta_{eta}_S_{S}.json')
    
                
                print(f"Bootstrap estimation {BOOT_IDX}", flush = True)
                
                #with open(opt_local_filename, 'r') as file:
                #    local_opt_result = json.load(file)
                    
                    
                #beta_local = local_opt_result['x']
                
                beta_local = [ 2.74173322,  7.88673881,  3.61469157, -7.68520778,  5.07195739,  0.88546586,
                  8.35367958,  1.6843873,   5.97599433,  1.51374912,  4.21720597, -9.6948498,
                  1.66753475 ,-7.22105927,  8.1818476,   2.70510033, 10.72818245, -2.31085699,
                  2.20399267]
                W_opt = estimator_empirical.compute_optimal_weighting_matrix(beta_local, empirical_moments, K_clients, K_politicians, eta, h, S, W)
               
                
                seed = BOOT_IDX
                rng = np.random.RandomState(seed)
            
                # C) Pairs Bootstrap: sample edges (rows) with replacement
                n_matches = len(matches_df)
                resample_indices = rng.randint(0, n_matches, size=n_matches)  # same size
                if BOOT_IDX == 1:
                    matches_df_boot = matches_df.copy()
                else:
                    matches_df.iloc[resample_indices].copy()
                
                empirical_moments_boot = compute_moments_from_data(matches_df_boot, buyers_structured, sellers_structured)
    
                eta = 100
                bounds = None
                seeds = None
                h='P'     
                S=1
                method = 'Nelder-Mead'
                options = {#'maxiter': 20,
                           'maxfev': 5,
                           'adaptive':True,
                           'disp': True}
                
    
                local_op_boot = estimator_empirical.estimate_betas_local(beta_local, empirical_moments_boot, K_clients, K_politicians, eta, h, S, W_opt, seeds, method, bounds, options)
                save_optimize_result_to_json(local_op_boot, opt_local_filename_boot, exclude_attrs = ['hess', 'hess_inv', 'final_simplex'])

            
    cont = False
    if cont:
        
        beta_initial_1 = [
        0.42277778986723225,
        8.397438505243393,
        2.4758871929752937,
        1.0789175016449073,
        -7.201248179753552,
        -7.3098017302428016,
        -7.609742694195727,
        -9.890017015838366,
        1.5347662486666327,
        8.461951818858044,
        2.9542671132336067,
        -7.393200956692693,
        -4.7886905286011725,
        4.176281501399838,
        4.934727316951584,
        0.3930656060231583,
        3.614386066875862,
        12.647274200305919,
        -1.7010342247598083
    ]
        
        beta_initial_2 = [  7.20730641,   0.88725614 ,-12.73926151, -12.34584751, -14.57958667,
 -14.11280548,   8.01467492,   4.4211512,    4.90394384,  -7.96739285,
 -10.19642059,   0.06838527, -13.06716831]
        
        beta_initial_3 = [ 0.0893019,   0.43895572,  2.89711199,  4.68229379, -0.62965384, -9.13991895,
 -6.59598607,  4.96553619,  3.70199025,  3.67500338, -4.01283334,  1.10132144,
  2.45290673,  4.74097523,  7.80943023,  4.35796321,  2.92999882,  0.87486231]
        
        beta_initial_4 = [
        0.4873841989710992,
        0.6573037186133179,
        2.1538424809424495,
        3.6626536045422675,
        -0.523877660562474,
        -3.4059150326223744,
        -3.3745740894385357,
        0.05790515129662133,
        -7.525503010781948,
        -5.392323257049032,
        -2.3269824552520326,
        5.05609685658219
    ]
        #beta_initial = beta_local
    
        error_distribution_clients= truncnorm(*get_truncnorm_params(0, -2, 2, 1))
        error_distribution_politicians = truncnorm(*get_truncnorm_params(0, -2, 2, 1))
        
        
        error_generator_buyers_params = (
            estimator_empirical.error_generator_buyers.dist.name,
            #estimator_empirical.error_generator_buyers.args
            error_distribution_clients.args
        )
        error_generator_sellers_params = (
            estimator_empirical.error_generator_sellers.dist.name,
            #estimator_empirical.error_generator_sellers.args
            error_distribution_politicians.args
        )
    
        beta_initial = beta_initial_4
        
        
       
        sims = {}
            
        for market_id in estimator_empirical.market_IDs:
            estimator_params = (
                estimator_empirical.buyers_data[estimator_empirical.buyers_data['Market']==market_id],
                estimator_empirical.sellers_data[estimator_empirical.sellers_data['Market']==market_id],
                error_generator_buyers_params,
                error_generator_sellers_params,
                K_clients,
                K_politicians,
                h  # Include h
            )
            
            args = (
                    42,
                    beta_initial[:K_clients],
                    beta_initial[K_clients:],
                    eta,
                    True,
                    'central',
                    'scott',
                    estimator_params,
                    None,
                    None
                )
            
            
                
            sims[market_id] = simulate_and_compute(args, return_simulation=True, compute_matches=True)
            
        #simulated_moments =  compute_moments_from_simulation(sim.buyers.data, sim.sellers.data, sim)
        
        beta_clients = beta_initial[:K_clients]
        beta_politicians = beta_initial[K_clients:]
        
        # Compute mean valuations
        p1, p2 = compute_and_plot_mean_valuation(sims, matches_df, beta_clients, beta_politicians, estimator_empirical.market_IDs)
        
        p1.savefig(machine + f"0119_model_all_years_fit_politicians_vendors_only_{vendors_only}_all_attributes_{all_attributes}.pdf", dpi=300)
        p2.savefig(machine + f"0119_model_all_years_fit_clients_vendors_only_{vendors_only}_all_attributes_{all_attributes}.pdf", dpi=300)
        
        #cont = False
        #if cont:
                   
        method = 'Nelder-Mead'
        options = {'maxiter': 100, 'adaptive':True, 'disp': True}
        bounds = None
        seeds = None
       
        #beta_initial =  [-0.72428274,  0.87587812, -0.77987981, -0.63690615,  0.13176297,  0.39015349, 0.26819727, -0.3212434,  -0.36535658]
        #beta_initial =  [-0.93738068,  1.63530396, -2.06593498, -2.57237813,  0.3085164,  0.78650522, 2.5341566, -3.76954783,  4.67150365]
        #beta_initial = [ 0.92570017, -3.33084337,  3.10318154,  2.59488348, -9.09288442, -4.20761132, 1.04524654]
        #beta_initial = [5.3554405,  -0.54586775,  4.82874899, -6.2309115,  -6.68625529,  0.04098387]
        # Run the estimati
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_local_1.log')
        estimator_empirical.set_logger(name='estimation_vendors_local_3', log_file_path=logger_path)
    
        
        result = estimator_empirical.estimate_betas_local(beta_initial, empirical_moments, K_clients, K_politicians, eta, h, S, W, seeds, method, bounds, options)
        
        # Extract the estimated beta values
        beta_estimated = result.x
        beta_buyers_estimated, beta_sellers_estimated = beta_estimated
        
        
        
        error_generator_buyers_params = (
            estimator_empirical.error_generator_buyers.dist.name,
            estimator_empirical.error_generator_buyers.args
        )
        error_generator_sellers_params = (
            estimator_empirical.error_generator_sellers.dist.name,
            estimator_empirical.error_generator_sellers.args
        )
    
    
        
        estimator_params = (
            estimator_empirical.buyers_data,
            estimator_empirical.sellers_data,
            error_generator_buyers_params,
            error_generator_sellers_params,
            K_clients,
            K_politicians,
            h  # Include h
        )
       
        
        args = (
                42,
                beta_initial[:K_clients],
                beta_initial[K_clients:],
                eta,
                True,
                'central',
                'scott',
                estimator_params,
                None,
                None
            )
            
        sim = simulate_and_compute(args, return_simulation=True, compute_matches=True)
        
        simulated_moments =  compute_moments_from_simulation(sim.buyers.data, sim.sellers.data, simulation)
        # Compute mean valuations
        
        
        beta_clients = beta_initial[:K_clients]
        beta_politicians = beta_initial[K_clients:]
        mean_valuation_dict = compute_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        
        # Compute mean valuations
        compute_and_plot_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        
        #######
        
        # ONLY VENDORS AND ALL ATTRIBUTES
    
        
        
        
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_global_3.log')
        estimator_empirical.set_logger(name='estimation_vendors_global_8', log_file_path=logger_path)
    
        bb = estimator_empirical.estimate_betas_global(beta_bounds, empirical_moments, K_clients, K_politicians, eta, h, S, W)
                        
        method = 'L-BFGS-B'
        options = {'maxiter': 100, 'adaptive':True, 'disp': True}
        options = {'maxiter': 100,  'disp': True}
        bounds = None
        seeds = None
       
        #beta_initial =  [-0.72428274,  0.87587812, -0.77987981, -0.63690615,  0.13176297,  0.39015349, 0.26819727, -0.3212434,  -0.36535658]
        #beta_initial =  [-0.93738068,  1.63530396, -2.06593498, -2.57237813,  0.3085164,  0.78650522, 2.5341566, -3.76954783,  4.67150365]
        beta_initial = [ 2.96260819, -3.77188555,  .79619947,  0.64993126, -9.26419684, -6.49298247, 2.05501951,  1.2469223,   2.53298759, -3.37055299,  1.40552219]
        #beta_initial =[ 0.92570017, -3.33084337,  3.10318154,  2.59488348, -9.09288442, -4.20761132, 1.04524654]
    
        # Run the estimati
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_local_2.log')
        estimator_empirical.set_logger(name='estimation_vendors_local_4', log_file_path=logger_path)
    
        
        result = estimator_empirical.estimate_betas_local(beta_initial, empirical_moments, K_clients, K_politicians, eta, h, S, W, seeds, method, bounds, options)
        
        # Extract the estimated beta values
        beta_estimated = result.x
        beta_buyers_estimated, beta_sellers_estimated = beta_estimated
        
        
        error_distribution_clients= truncnorm(*get_truncnorm_params(0, -1, 1, 1))
        error_distribution_politicians = truncnorm(*get_truncnorm_params(0, -1, 1, 1))
        
        error_generator_buyers_params = (
            estimator_empirical.error_generator_buyers.dist.name,
            #estimator_empirical.error_generator_buyers.args
            error_distribution_clients.args
        )
        error_generator_sellers_params = (
            estimator_empirical.error_generator_sellers.dist.name,
            #estimator_empirical.error_generator_sellers.args
            error_distribution_politicians.args
        )
    
    
        
        estimator_params = (
            estimator_empirical.buyers_data,
            estimator_empirical.sellers_data,
            error_generator_buyers_params,
            error_generator_sellers_params,
            K_clients,
            K_politicians,
            h  # Include h
        )
       
        
        args = (
                42,
                beta_initial[:K_clients],
                beta_initial[K_clients:],
                eta,
                True,
                'central',
                'scott',
                estimator_params,
                None,
                None
            )
            
        sim = simulate_and_compute(args, return_simulation=True, compute_matches=True)
        
        simulated_moments =  compute_moments_from_simulation(sim.buyers.data, sim.sellers.data, sim)
        
        beta_clients = beta_initial[:K_clients]
        beta_politicians = beta_initial[K_clients:]
        
        # Compute mean valuations
        p1, p2 = compute_and_plot_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        p1.savefig("model_fit_politicians.pdf", dpi=300)
        p2.savefig("model_fit_clients.pdf", dpi=300)
        #######
        
        # ALL CLIENTS AND SOME ATTRIBUTES
        
        
        
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_global_10.log')
        estimator_empirical.set_logger(name='estimation_vendors_global_10', log_file_path=logger_path)
    
        bb = estimator_empirical.estimate_betas_global(beta_bounds, empirical_moments, K_clients, K_politicians, eta, h, S, W)
                        
        method = 'L-BFGS-B'
        options = {'maxiter': 100, 'adaptive':True, 'disp': True}
        options = {'maxiter': 100,  'disp': True}
        bounds = None
        seeds = None
       
        #beta_initial =  [-0.72428274,  0.87587812, -0.77987981, -0.63690615,  0.13176297,  0.39015349, 0.26819727, -0.3212434,  -0.36535658]
        #beta_initial =  [-0.93738068,  1.63530396, -2.06593498, -2.57237813,  0.3085164,  0.78650522, 2.5341566, -3.76954783,  4.67150365]
        beta_initial =[ 0.92570017, -3.33084337,  3.10318154,  2.59488348, -9.09288442, -4.20761132, 1.04524654]
        beta_initial = [5.3554405,  -0.54586775,  4.82874899, -5.2309115,  -5.68625529,  0.34098387]
    
        # Run the estimati
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_local_1.log')
        estimator_empirical.set_logger(name='estimation_vendors_local_3', log_file_path=logger_path)
    
        
        result = estimator_empirical.estimate_betas_local(beta_initial, empirical_moments, K_clients, K_politicians, eta, h, S, W, seeds, method, bounds, options)
        
        # Extract the estimated beta values
        beta_estimated = result.x
        beta_buyers_estimated, beta_sellers_estimated = beta_estimated
        
        
        
        error_generator_buyers_params = (
            estimator_empirical.error_generator_buyers.dist.name,
            estimator_empirical.error_generator_buyers.args
        )
        error_generator_sellers_params = (
            estimator_empirical.error_generator_sellers.dist.name,
            estimator_empirical.error_generator_sellers.args
        )
    
    
        
        estimator_params = (
            estimator_empirical.buyers_data,
            estimator_empirical.sellers_data,
            error_generator_buyers_params,
            error_generator_sellers_params,
            K_clients,
            K_politicians,
            h  # Include h
        )
       
        
        args = (
                42,
                beta_initial[:K_clients],
                beta_initial[K_clients:],
                eta,
                True,
                'central',
                'scott',
                estimator_params,
                None,
                None
            )
            
        sim = simulate_and_compute(args, return_simulation=True, compute_matches=True)
        
        simulated_moments =  compute_moments_from_simulation(sim.buyers.data, sim.sellers.data, simulation)
        # Compute mean valuations
        
        
        beta_clients = beta_initial[:K_clients]
        beta_politicians = beta_initial[K_clients:]
        mean_valuation_dict = compute_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        
        # Compute mean valuations
        compute_and_plot_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        
        
        
        
        
        # ALL CLIENTS AND ALL ATTRIBUTES
        
        
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_global_10.log')
        estimator_empirical.set_logger(name='estimation_vendors_global_10', log_file_path=logger_path)
    
        bb = estimator_empirical.estimate_betas_global(beta_bounds, empirical_moments, K_clients, K_politicians, eta, h, S, W)
                        
        method = 'L-BFGS-B'
        options = {'maxiter': 100, 'adaptive':True, 'disp': True}
        options = {'maxiter': 100,  'disp': True}
        bounds = None
        seeds = None
       
        #beta_initial =  [-0.72428274,  0.87587812, -0.77987981, -0.63690615,  0.13176297,  0.39015349, 0.26819727, -0.3212434,  -0.36535658]
        #beta_initial =  [-0.93738068,  1.63530396, -2.06593498, -2.57237813,  0.3085164,  0.78650522, 2.5341566, -3.76954783,  4.67150365]
        beta_initial =[ 0.92570017, -3.33084337,  3.10318154,  2.59488348, -9.09288442, -4.20761132, 1.04524654]
        beta_initial = [5.3554405,  -0.54586775,  4.82874899, -5.2309115,  -5.68625529,  0.34098387]
    
    
        #beta_initial = [ 3.97633815 -2.59260512  6.62255997 -7.07844909 -2.59107032  5.75252563,1.05324422 , 9.34098669, -4.20041562,  9.09829059]
        
        # Run the estimati
        # Before calling the estimator, set a logger
        logger_path=os.path.join(machine,'estimation_logs/estimation_vendors_local_1.log')
        estimator_empirical.set_logger(name='estimation_vendors_local_3', log_file_path=logger_path)
    
        
        result = estimator_empirical.estimate_betas_local(beta_initial, empirical_moments, K_clients, K_politicians, eta, h, S, W, seeds, method, bounds, options)
        
        # Extract the estimated beta values
        beta_estimated = result.x
        beta_buyers_estimated, beta_sellers_estimated = beta_estimated
        
        
        
        error_generator_buyers_params = (
            estimator_empirical.error_generator_buyers.dist.name,
            estimator_empirical.error_generator_buyers.args
        )
        error_generator_sellers_params = (
            estimator_empirical.error_generator_sellers.dist.name,
            estimator_empirical.error_generator_sellers.args
        )
    
    
        
        estimator_params = (
            estimator_empirical.buyers_data,
            estimator_empirical.sellers_data,
            error_generator_buyers_params,
            error_generator_sellers_params,
            K_clients,
            K_politicians,
            h  # Include h
        )
       
        
        args = (
                42,
                beta_initial[:K_clients],
                beta_initial[K_clients:],
                eta,
                True,
                'central',
                'scott',
                estimator_params,
                None,
                None
            )
            
        sim = simulate_and_compute(args, return_simulation=True, compute_matches=True)
        
        simulated_moments =  compute_moments_from_simulation(sim.buyers.data, sim.sellers.data, simulation)
        # Compute mean valuations
        
        
        beta_clients = beta_initial[:K_clients]
        beta_politicians = beta_initial[K_clients:]
        mean_valuation_dict = compute_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        
        # Compute mean valuations
        compute_and_plot_mean_valuation(sim, matches_df, beta_clients, beta_politicians)
        
        
def main():
    # macOS and Windows require freeze_support
    
    empirical_estimation()

if __name__ == "__main__":
    main()
    
def plotting():
        
    
    eta=100
    S=5
    
    all_attributes = True
    vendors_only = True
        
    opt_local_filename = os.path.join(machine, f'estimation_results/local_estimation_vendors_only_{vendors_only}_all_attributes_{all_attributes}_eta_{eta}_S_{S}.json')
    with open(opt_local_filename, 'r') as file:
        est = json.load(file)['x']
    
    N_boot = 100
    est_boot_array = np.zeros((len(est),N_boot))
    for BOOT_IDX in range(1,N_boot+1):
        opt_local_filename_boot = os.path.join(machine, f'estimation_results/local_estimation_boot_{BOOT_IDX}_vendors_only_{vendors_only}_all_attributes_{all_attributes}_eta_{eta}_S_{S}.json')
        with open(opt_local_filename_boot, 'r') as file:
            est_boot_array[:,BOOT_IDX-1] = json.load(file)['x']

    #estimate = np.array(est)
    #lower_ci = np.percentile(est_boot_array, 2.5, axis=1)
    #upper_ci = np.percentile(est_boot_array, 97.5, axis=1)
    
    estimate = np.array(beta_initial_1)
    
    estimate = np.array(beta_initial_1)
    variables = ['is_retained', 'log_asset_norm',
                 'connection_lobbyist', 'revolving_flag', 'is_chair', 'appropriations_committee', 'senator', 'minority_party', 'cfscore_con', 'GOVT', 'share_votes_general']
    example_data_model2 = pd.DataFrame({
        'Model': list(np.repeat('Complete',11)),
        'VariableName': variables,
        'VariableGroup': list(np.repeat('Clients',2)) + list(np.repeat('Lawmakers',9)),
        'VariableNameToDisplay': [
            'Retained lobbyist', 'Log assets (normalized)',
            'Has PAC connection', 'Has revolving-flag connection', 'Chair committee', 'Appropriations committee', 'Senator', 'Minority party', 'CF score', 'Previous gov.t exp.', 'Share votes'
        ],
        'estimate': list(estimate[:2])+list(estimate[6:15]),
        'lower_ci': list(lower_ci[:2])+list(lower_ci[6:15]),
        'upper_ci': list(upper_ci[:2])+list(upper_ci[6:15])
    })
    
    all_attributes = False
    vendors_only = True
    
    estimate = np.array(beta_initial_2)
    variables = ['is_retained', 'log_asset_norm',
                 'connection_lobbyist', 'revolving_flag', 'LES']
    # Example usage with two models
    example_data_model1 = pd.DataFrame({
        'Model': list(np.repeat('Reduced',5)),
        'VariableName': variables,
        'VariableGroup': list(np.repeat('Clients',2)) + list(np.repeat('Lawmakers',3)),
        'VariableNameToDisplay': [
            'Retained lobbyist', 'Log assets (normalized)', 
            'Has PAC connection', 'Has revolving-flag connection', 'LES'
        ],
        'estimate': list(estimate[:2])+list(estimate[6:9]),
        'lower_ci': list(lower_ci[:2])+list(lower_ci[6:9]),
        'upper_ci': list(upper_ci[:2])+list(upper_ci[6:9])
    })
    
    create_multi_model_forest_plot([example_data_model2], save_path='/Users/michelev/0119_vendors_forest_plot_0107.pdf')
    
    all_attributes = False
    vendors_only = True
    # Example usage with two models
    
    estimate = np.array(beta_initial_4)
    
    example_data_model3 = pd.DataFrame({
        'Model': list(np.repeat('Reduced',4)),
        'VariableName': ['is_retained',
                         'connection_lobbyist', 'revolving_flag', 'LES'],
        'VariableGroup': list(np.repeat('Clients',1)) + list(np.repeat('Lawmakers',3)),
        'VariableNameToDisplay': [
             'Retained lobbyist', 
             'Has PAC connection', 'Has revolving-flag connection', 'LES'
        ],
        'estimate': list(estimate[:1])+list(estimate[5:8]),
        'lower_ci': list(lower_ci[:1])+list(lower_ci[5:8]),
        'upper_ci': list(upper_ci[:1])+list(upper_ci[5:8])
    })
    
    all_attributes = False
    vendors_only = False
    
    estimate = np.array(beta_initial_3)
    
    example_data_model4 = pd.DataFrame({
        'Model': list(np.repeat('Complete',10)),
        'VariableName': ['is_retained', 
                         'connection_lobbyist', 'revolving_flag', 'is_chair', 'appropriations_committee',  'senator', 'minority_party', 'cfscore_con', 'GOVT', 'share_votes_general'],
        'VariableGroup': list(np.repeat('Clients',1)) + list(np.repeat('Lawmakers',9)),
        'VariableNameToDisplay': [
            'Retained lobbyist',
            'Has PAC connection', 'Has revolving-flag connection', 'Chair committee', 'Appropriations committee', 'Senator', 'Minority party', 'CF score', 'Previous gov.t exp.', 'Share votes'
        ],
        'estimate': list(estimate[:1])+list(estimate[5:14]),
        'lower_ci': list(lower_ci[:1])+list(lower_ci[5:14]),
        'upper_ci': list(upper_ci[:1])+list(upper_ci[5:14])
    })
    
    
    create_multi_model_forest_plot([example_data_model4], save_path='/Users/michelev/0119_vendors_forest_plot_all_0107.pdf')
    
    



#if __name__ == "__main__":
#    main()
