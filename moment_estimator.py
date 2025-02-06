from scipy.stats import truncnorm
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from market_simulation import MarketSide, MarketSimulation
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize, differential_evolution
import warnings
import logging
import multiprocessing
import os
import random
from Base import get_truncnorm_range, generate_predictors, create_error_generator, perturb_duplicates

#%%

base_seed = 42  # Set a base seed for reproducibility


def get_allocated_cpus():
    """
    Retrieve the number of CPUs allocated to this Slurm task.
    """
    slurm_cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus_per_task:
        try:
            # SLURM_CPUS_PER_TASK can sometimes be in the format '2(x5)'
            return int(slurm_cpus_per_task.split('(')[0])
        except (ValueError, AttributeError):
            pass
    # Fallback to the number of CPUs available on the machine
    return multiprocessing.cpu_count()


available_cpus = get_allocated_cpus()

def get_logger(name, log_file_path=None, level=logging.INFO):
    """
    Helper function to create or retrieve a logger by name, optionally with its own file handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent logging from propagating to the root logger
    
    # Check if the logger already has handlers (to avoid adding duplicates)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # If a log_file_path is given, log to file, else log to console
        if log_file_path:
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    return logger

def truncated_normal(mean=0, sd=1, low=0, upp=10, size=1):
    """
    Generate samples from a truncated normal distribution.
    
    :param mean: Mean of the underlying normal distribution.
    :param sd: Standard deviation of the underlying normal distribution.
    :param low: Lower bound of the truncation.
    :param upp: Upper bound of the truncation.
    :param size: Number of samples to generate.
    :return: Array of truncated normal samples.
    """
    a, b = (low - mean) / sd, (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd).rvs(size)
    
def compute_moments_from_data(matches_df, buyers_data, sellers_data):

    """
    Compute the moments based on the actual observed matches.

    Each buyer's moment includes:
      - Outer product of the buyer's predictor vector with the average of their matched sellers' predictor vectors.
      - Outer product of the buyer's predictor vector with the std dev of their matched sellers' predictor vectors.

    Similarly, each seller's moment includes:
      - Outer product of the seller's predictor vector with the average of their matched buyers' predictor vectors.
      - Outer product of the seller's predictor vector with the std dev of their matched buyers' predictor vectors.

    Both average and std moments are stored along a new first dimension:
      - 0: Mean moments
      - 1: Std moments

    Additionally, moments for unmatched buyers and sellers are computed and included.

    Parameters:
    - matches_df (pd.DataFrame): DataFrame containing matches with columns 'buyer_ID' and 'seller_ID'.
    - buyers_data (np.ndarray): Structured array with fields 'ID', 'Predictors'.
    - sellers_data (np.ndarray): Structured array with fields 'ID', 'Predictors'.

    Returns:
    - dict: {
        "moment_buyers": np.ndarray of shape (2, K_buyers, K_sellers),
        "moment_sellers": np.ndarray of shape (2, K_sellers, K_buyers),
        "moment_unmatched_buyers": np.ndarray of shape (2, K_buyers),
        "moment_unmatched_sellers": np.ndarray of shape (2, K_sellers),
      }
    """
    
    market_IDs = np.unique(buyers_data['Market'])
    
    # Number of predictors for buyers and sellers
    K_buyers = buyers_data['Predictors'].shape[1]
    K_sellers = sellers_data['Predictors'].shape[1]
    
    moments = {}
    if len(market_IDs)>1:
        
        for market_ID in market_IDs:
            
            buyers_data_market = buyers_data[buyers_data['Market'] == market_ID]
            sellers_data_market = sellers_data[sellers_data['Market'] == market_ID]
            matches_df_market = matches_df[matches_df['Market'] == market_ID]
            
        
            # Create mappings from IDs to indices for quick access
            buyer_id_to_index = {buyer['ID']: idx for idx, buyer in enumerate(buyers_data_market)}
            seller_id_to_index = {seller['ID']: idx for idx, seller in enumerate(sellers_data_market)}
            
            
            # Initialize lists to collect outer products
            outer_products_buyers_avg = []
            outer_products_buyers_std = []
            outer_products_sellers_avg = []
            outer_products_sellers_std = []
            
            # Lists to track unmatched buyers and sellers
            # To find unmatched, first find all matched buyers and sellers
            matched_buyers = set(matches_df_market['buyer_ID'].unique())
            matched_sellers = set(matches_df_market['seller_ID'].unique())
            
            all_buyers = set(buyers_data_market['ID'])
            all_sellers = set(sellers_data_market['ID'])
            
            unmatched_buyers = all_buyers - matched_buyers
            unmatched_sellers = all_sellers - matched_sellers
            
            # Extract buyers' predictor data
            buyers_predictors = buyers_data_market['Predictors']
            sellers_predictors = sellers_data_market['Predictors']
            
            # Create a dictionary mapping buyer_ID to list of seller_IDs
            buyers_matches = matches_df_market.groupby('buyer_ID')['seller_ID'].apply(list).to_dict()
            
            # Create a dictionary mapping seller_ID to list of buyer_IDs
            sellers_matches = matches_df_market.groupby('seller_ID')['buyer_ID'].apply(list).to_dict()
            
            unmatched_buyers_predictors = []
            unmatched_sellers_predictors = []
            # --- Compute Moments from Buyers' Matches ---
            for buyer_id, matched_sellers_IDs in buyers_matches.items():
                if len(matched_sellers_IDs) == 0:
                    # Collect predictors of unmatched buyers
                    if buyer_id in buyer_id_to_index:
                        unmatched_buyers_predictors.append(buyers_predictors[buyer_id_to_index[buyer_id]])
                    continue  # Skip buyers with no matches
        
                # Ensure the buyer ID exists
                if buyer_id not in buyer_id_to_index:
                    warnings.warn(f"Buyer ID {buyer_id} not found in buyers_data. Skipping.")
                    continue
        
                buyer_idx = buyer_id_to_index[buyer_id]
                x_buyer = buyers_predictors[buyer_idx,:-4]  # Shape: (K_buyers,)
        
                # Collect matched sellers' predictor vectors
                matched_sellers_indices = []
                for seller_id in matched_sellers_IDs:
                    if seller_id not in seller_id_to_index:
                        warnings.warn(f"Seller ID {seller_id} not found in sellers_data. Skipping.")
                        continue
                    matched_sellers_indices.append(seller_id_to_index[seller_id])
        
                if len(matched_sellers_indices) == 0:
                    # No valid matched sellers found
                    unmatched_buyers_predictors.append(x_buyer)
                    continue
        
                x_sellers_matched = sellers_predictors[matched_sellers_indices,:-4]  # Shape: (S_b, K_sellers)
        
                # Compute the average and std dev of matched sellers' predictors
                average_x_sellers = np.mean(x_sellers_matched, axis=0)             # Shape: (K_sellers,)
                std_x_sellers = np.std(x_sellers_matched, axis=0, ddof=0)          # Shape: (K_sellers,)
        
                # Compute the outer products
                outer_product_avg = np.outer(x_buyer, average_x_sellers)  # Shape: (K_buyers, K_sellers)
                outer_product_std = np.outer(x_buyer, std_x_sellers)      # Shape: (K_buyers, K_sellers)
        
                # Append to the lists
                outer_products_buyers_avg.append(outer_product_avg)
                outer_products_buyers_std.append(outer_product_std)
        
            # --- Compute Moments from Sellers' Matches ---
            for seller_id, matched_buyers_IDs in sellers_matches.items():
                if len(matched_buyers_IDs) == 0:
                    # Collect predictors of unmatched sellers
                    if seller_id in seller_id_to_index:
                        unmatched_sellers_predictors.append(sellers_predictors[seller_id_to_index[seller_id]])
                    continue  # Skip sellers with no matches
        
                # Ensure the seller ID exists
                if seller_id not in seller_id_to_index:
                    warnings.warn(f"Seller ID {seller_id} not found in sellers_data. Skipping.")
                    continue
        
                seller_idx = seller_id_to_index[seller_id]
                x_seller = sellers_predictors[seller_idx,:-4]  # Shape: (K_sellers,)
        
                # Collect matched buyers' predictor vectors
                matched_buyers_indices = []
                for buyer_id in matched_buyers_IDs:
                    if buyer_id not in buyer_id_to_index:
                        warnings.warn(f"Buyer ID {buyer_id} not found in buyers_data. Skipping.")
                        continue
                    matched_buyers_indices.append(buyer_id_to_index[buyer_id])
        
                if len(matched_buyers_indices) == 0:
                    # No valid matched buyers found
                    unmatched_sellers_predictors.append(x_seller)
                    continue
        
                x_buyers_matched = buyers_predictors[matched_buyers_indices,:-4]  # Shape: (B_s, K_buyers)
        
                # Compute the average and std dev of matched buyers' predictors
                average_x_buyers = np.mean(x_buyers_matched, axis=0)             # Shape: (K_buyers,)
                std_x_buyers = np.std(x_buyers_matched, axis=0, ddof=0)          # Shape: (K_buyers,)
        
                # Compute the outer products
                outer_product_avg = np.outer(x_seller, average_x_buyers)  # Shape: (K_sellers, K_buyers)
                outer_product_std = np.outer(x_seller, std_x_buyers)      # Shape: (K_sellers, K_buyers)
        
                # Append to the lists
                outer_products_sellers_avg.append(outer_product_avg)
                outer_products_sellers_std.append(outer_product_std)
        
            # --- Aggregate and Compute Average Moments ---
            # Buyers' Moments
            if len(outer_products_buyers_avg) > 0:
                sum_outer_products_buyers_avg = np.sum(outer_products_buyers_avg, axis=0)  # Shape: (K_buyers, K_sellers)
                moment_buyers_avg = sum_outer_products_buyers_avg / len(outer_products_buyers_avg)  # Shape: (K_buyers, K_sellers)
        
                sum_outer_products_buyers_std = np.sum(outer_products_buyers_std, axis=0)  # Shape: (K_buyers, K_sellers)
                moment_buyers_std = sum_outer_products_buyers_std / len(outer_products_buyers_std)  # Shape: (K_buyers, K_sellers)
            else:
                warnings.warn("No buyer matches found. Setting moment_buyers_avg and moment_buyers_std to zero matrices.")
                moment_buyers_avg = np.zeros((K_buyers-4, K_sellers-4))
                moment_buyers_std = np.zeros((K_buyers-4, K_sellers-4))
        
            moment_buyers = np.stack([moment_buyers_avg, moment_buyers_std], axis=0)  # Shape: (2, K_buyers, K_sellers)
        
            # Sellers' Moments
            if len(outer_products_sellers_avg) > 0:
                sum_outer_products_sellers_avg = np.sum(outer_products_sellers_avg, axis=0)  # Shape: (K_sellers, K_buyers)
                moment_sellers_avg = sum_outer_products_sellers_avg / len(outer_products_sellers_avg)  # Shape: (K_sellers, K_buyers)
        
                sum_outer_products_sellers_std = np.sum(outer_products_sellers_std, axis=0)  # Shape: (K_sellers, K_buyers)
                moment_sellers_std = sum_outer_products_sellers_std / len(outer_products_sellers_std)  # Shape: (K_sellers, K_buyers)
            else:
                warnings.warn("No seller matches found. Setting moment_sellers_avg and moment_sellers_std to zero matrices.")
                moment_sellers_avg = np.zeros((K_sellers-4, K_buyers-4))
                moment_sellers_std = np.zeros((K_sellers-4, K_buyers-4))
        
            moment_sellers = np.stack([moment_sellers_avg, moment_sellers_std], axis=0)  # Shape: (2, K_sellers, K_buyers)
        
            # --- Compute Moments for Unmatched Buyers ---
            if len(unmatched_buyers) > 0:
                # Filter unmatched_buyers to ensure they exist in buyers_data
                valid_unmatched_buyers = [buyer_id for buyer_id in unmatched_buyers if buyer_id in buyer_id_to_index]
                if len(valid_unmatched_buyers) > 0:
                    x_unmatched_buyers = buyers_predictors[[buyer_id_to_index[bid] for bid in valid_unmatched_buyers]]  # Shape: (N_unmatched_buyers, K_buyers)
                    x_unmatched_buyers = x_unmatched_buyers[:,:-4]
                    avg_unmatched_buyers = np.mean(x_unmatched_buyers, axis=0)  # Shape: (K_buyers,)
                    std_unmatched_buyers = np.std(x_unmatched_buyers, axis=0, ddof=0)  # Shape: (K_buyers,)
                else:
                    warnings.warn("No valid unmatched buyers found. Setting to zero.")
                    avg_unmatched_buyers = np.zeros(K_buyers-4)
                    std_unmatched_buyers = np.zeros(K_buyers-4)
            else:
                warnings.warn("No unmatched buyers found. Setting moment_unmatched_buyers_avg and moment_unmatched_buyers_std to zero vectors.")
                avg_unmatched_buyers = np.zeros(K_buyers-4)
                std_unmatched_buyers = np.zeros(K_buyers-4)
        
            moment_unmatched_buyers = np.stack([avg_unmatched_buyers, std_unmatched_buyers], axis=0)  # Shape: (2, K_buyers)
        
            # --- Compute Moments for Unmatched Sellers ---
            if len(unmatched_sellers) > 0:
                # Filter unmatched_sellers to ensure they exist in sellers_data
                valid_unmatched_sellers = [seller_id for seller_id in unmatched_sellers if seller_id in seller_id_to_index]
                if len(valid_unmatched_sellers) > 0:
                    x_unmatched_sellers = sellers_predictors[[seller_id_to_index[sid] for sid in valid_unmatched_sellers]]  # Shape: (N_unmatched_sellers, K_sellers)
                    x_unmatched_sellers = x_unmatched_sellers[:,:-4]
                    avg_unmatched_sellers = np.mean(x_unmatched_sellers, axis=0)  # Shape: (K_sellers,)
                    std_unmatched_sellers = np.std(x_unmatched_sellers, axis=0, ddof=0)  # Shape: (K_sellers,)
                else:
                    warnings.warn("No valid unmatched sellers found. Setting to zero.")
                    avg_unmatched_sellers = np.zeros(K_sellers-4)
                    std_unmatched_sellers = np.zeros(K_sellers-4)
            else:
                warnings.warn("No unmatched sellers found. Setting moment_unmatched_sellers_avg and moment_unmatched_sellers_std to zero vectors.")
                avg_unmatched_sellers = np.zeros(K_sellers-4)
                std_unmatched_sellers = np.zeros(K_sellers-4)
        
            moment_unmatched_sellers = np.stack([avg_unmatched_sellers, std_unmatched_sellers], axis=0)  # Shape: (2, K_sellers)
        
            # Store all moments in a dictionary
            moments_market = {
                "moment_buyers": moment_buyers,                       # Shape: (2, K_buyers, K_sellers)
                "moment_sellers": moment_sellers,                     # Shape: (2, K_sellers, K_buyers)
                "moment_unmatched_buyers": moment_unmatched_buyers,   # Shape: (2, K_buyers)
                "moment_unmatched_sellers": moment_unmatched_sellers  # Shape: (2, K_sellers)
            }
            
            moments[market_ID] = moments_market

    return moments

def compute_moments_from_simulation(buyers_data, sellers_data, simulation):
    """
    Compute the moments based on the simulated equilibrium.

    Each buyer's moment includes:
      - Outer product of the buyer's predictor vector with the average of their matched sellers' predictor vectors.
      - Outer product of the buyer's predictor vector with the std dev of their matched sellers' predictor vectors.

    Similarly, each seller's moment includes:
      - Outer product of the seller's predictor vector with the average of their matched buyers' predictor vectors.
      - Outer product of the seller's predictor vector with the std dev of their matched buyers' predictor vectors.

    Both average and std moments are stored along a new first dimension:
      - 0: Mean moments
      - 1: Std moments

    Additionally, moments for unmatched buyers and sellers are computed and included.

    Parameters:
    - buyers_data (np.ndarray): Structured array with fields 'ID', 'Predictors', 'Valuation'.
    - sellers_data (np.ndarray): Structured array with fields 'ID', 'Predictors', 'Valuation'.
    - simulation (MarketSimulation): Object containing 'buyers_matches' and 'sellers_matches' dictionaries.

    Returns:
    - dict: {
        "moment_buyers": np.ndarray of shape (2, K_buyers, K_sellers),
        "moment_sellers": np.ndarray of shape (2, K_sellers, K_buyers),
        "moment_unmatched_buyers": np.ndarray of shape (2, K_buyers),
        "moment_unmatched_sellers": np.ndarray of shape (2, K_sellers),
      }
    """
    # Create mappings from IDs to indices for quick access
    buyer_id_to_index = {id: idx for idx, id in enumerate(buyers_data['ID'])}
    seller_id_to_index = {id: idx for idx, id in enumerate(sellers_data['ID'])}

    # Number of predictors for buyers and sellers
    K_buyers = buyers_data['Predictors'].shape[1]
    K_sellers = sellers_data['Predictors'].shape[1]

    # Initialize lists to collect outer products
    outer_products_buyers_avg = []
    outer_products_buyers_std = []
    outer_products_sellers_avg = []
    outer_products_sellers_std = []

    # Lists to track unmatched buyers and sellers
    unmatched_buyers_predictors = []
    unmatched_sellers_predictors = []

    # --- Compute Moments from Buyers' Matches ---
    matched_buyers_count = 0
    for buyer_id, matched_sellers_IDs in simulation.buyers_matches.items():
        if len(matched_sellers_IDs) == 0:
            # Collect predictors of unmatched buyers
            buyer_idx = buyer_id_to_index.get(buyer_id, None)
            if buyer_idx is not None:
                unmatched_buyers_predictors.append(buyers_data['Predictors'][buyer_idx])
            continue  # Skip buyers with no matches

        matched_buyers_count += 1

        # Ensure the buyer ID exists
        if buyer_id not in buyer_id_to_index:
            raise ValueError(f"Buyer ID {buyer_id} not found in buyers_data.")

        buyer_idx = buyer_id_to_index[buyer_id]
        x_buyer = buyers_data['Predictors'][buyer_idx,:-4]  # Shape: (K_buyers,)

        # Collect matched sellers' predictor vectors
        matched_sellers_indices = []
        for seller_id in matched_sellers_IDs:
            if seller_id not in seller_id_to_index:
                raise ValueError(f"Seller ID {seller_id} not found in sellers_data.")
            matched_sellers_indices.append(seller_id_to_index[seller_id])

        x_sellers_matched = sellers_data['Predictors'][matched_sellers_indices,:-4]  # Shape: (S_b, K_sellers)

        # Compute the average and std dev of matched sellers' predictors
        average_x_sellers = np.mean(x_sellers_matched, axis=0)  # Shape: (K_sellers,)
        #average_x_sellers = np.sum(x_sellers_matched, axis=0)  # Shape: (K_sellers,)
        std_x_sellers = np.std(x_sellers_matched, axis=0, ddof=0)  # Shape: (K_sellers,)
        #std_x_sellers = np.sqrt(np.sum((x_sellers_matched-average_x_sellers)**2, axis=0))  # Shape: (K_sellers,)

        # Compute the outer products
        outer_product_avg = np.outer(x_buyer, average_x_sellers)  # Shape: (K_buyers, K_sellers)
        outer_product_std = np.outer(x_buyer, std_x_sellers)      # Shape: (K_buyers, K_sellers)

        # Append to the lists
        outer_products_buyers_avg.append(outer_product_avg)
        outer_products_buyers_std.append(outer_product_std)

    # --- Compute Moments from Sellers' Matches ---
    matched_sellers_count = 0
    for seller_id, matched_buyers_IDs in simulation.sellers_matches.items():
        if len(matched_buyers_IDs) == 0:
            # Collect predictors of unmatched sellers
            seller_idx = seller_id_to_index.get(seller_id, None)
            if seller_idx is not None:
                unmatched_sellers_predictors.append(sellers_data['Predictors'][seller_idx])
            continue  # Skip sellers with no matches

        matched_sellers_count += 1

        # Ensure the seller ID exists
        if seller_id not in seller_id_to_index:
            raise ValueError(f"Seller ID {seller_id} not found in sellers_data.")

        seller_idx = seller_id_to_index[seller_id]
        x_seller = sellers_data['Predictors'][seller_idx,:-4]  # Shape: (K_sellers,)

        # Collect matched buyers' predictor vectors
        matched_buyers_indices = []
        for buyer_id in matched_buyers_IDs:
            if buyer_id not in buyer_id_to_index:
                raise ValueError(f"Buyer ID {buyer_id} not found in buyers_data.")
            matched_buyers_indices.append(buyer_id_to_index[buyer_id])

        x_buyers_matched = buyers_data['Predictors'][matched_buyers_indices,:-4]  # Shape: (B_s, K_buyers)

        average_x_buyers = np.mean(x_buyers_matched, axis=0)  # Shape: (K_sellers,)
        #average_x_buyers = np.sum(x_buyers_matched, axis=0)  # Shape: (K_sellers,)
        std_x_buyers = np.std(x_buyers_matched, axis=0, ddof=0)  # Shape: (K_sellers,)
        #std_x_buyers = np.sqrt(np.sum(x_buyers_matched-average_x_buyers)**2, axis=0) Shape: (K_sellers,)

        # Compute the outer products
        outer_product_avg = np.outer(x_seller, average_x_buyers)  # Shape: (K_sellers, K_buyers)
        outer_product_std = np.outer(x_seller, std_x_buyers)      # Shape: (K_sellers, K_buyers)

        # Append to the lists
        outer_products_sellers_avg.append(outer_product_avg)
        outer_products_sellers_std.append(outer_product_std)

    # --- Aggregate and Compute Average Moments ---
    # Buyers' Moments
    if matched_buyers_count > 0:
        sum_outer_products_buyers_avg = np.sum(outer_products_buyers_avg, axis=0)  # Shape: (K_buyers, K_sellers)
        moment_buyers_avg = sum_outer_products_buyers_avg / matched_buyers_count  # Shape: (K_buyers, K_sellers)

        sum_outer_products_buyers_std = np.sum(outer_products_buyers_std, axis=0)  # Shape: (K_buyers, K_sellers)
        moment_buyers_std = sum_outer_products_buyers_std / matched_buyers_count  # Shape: (K_buyers, K_sellers)
    else:
        warnings.warn("No buyer matches found. Setting moment_buyers_avg and moment_buyers_std to zero matrices.")
        moment_buyers_avg = np.zeros((K_buyers-4, K_sellers-4))
        moment_buyers_std = np.zeros((K_buyers-4, K_sellers-4))

    moment_buyers = np.stack([moment_buyers_avg, moment_buyers_std], axis=0)  # Shape: (2, K_buyers, K_sellers)

    # Sellers' Moments
    if matched_sellers_count > 0:
        sum_outer_products_sellers_avg = np.sum(outer_products_sellers_avg, axis=0)  # Shape: (K_sellers, K_buyers)
        moment_sellers_avg = sum_outer_products_sellers_avg / matched_sellers_count  # Shape: (K_sellers, K_buyers)

        sum_outer_products_sellers_std = np.sum(outer_products_sellers_std, axis=0)  # Shape: (K_sellers, K_buyers)
        moment_sellers_std = sum_outer_products_sellers_std / matched_sellers_count  # Shape: (K_sellers, K_buyers)
    else:
        warnings.warn("No seller matches found. Setting moment_sellers_avg and moment_sellers_std to zero matrices.")
        moment_sellers_avg = np.zeros((K_sellers-4, K_buyers-4))
        moment_sellers_std = np.zeros((K_sellers-4, K_buyers-4))

    moment_sellers = np.stack([moment_sellers_avg, moment_sellers_std], axis=0)  # Shape: (2, K_sellers, K_buyers)

    # --- Compute Moments for Unmatched Buyers ---
    if unmatched_buyers_predictors:
        x_unmatched_buyers = np.array(unmatched_buyers_predictors)  # Shape: (N_unmatched_buyers, K_buyers)
        x_unmatched_buyers = x_unmatched_buyers[:,:-4]   # Shape: (N_unmatched_buyers, K_buyers)
 
        avg_unmatched_buyers = np.mean(x_unmatched_buyers, axis=0)  # Shape: (K_buyers,)
        std_unmatched_buyers = np.std(x_unmatched_buyers, axis=0, ddof=0)  # Shape: (K_buyers,)

        moment_unmatched_buyers_avg = avg_unmatched_buyers  # Shape: (K_buyers,)
        moment_unmatched_buyers_std = std_unmatched_buyers  # Shape: (K_buyers,)
    else:
        warnings.warn("No unmatched buyers found. Setting moment_unmatched_buyers_avg and moment_unmatched_buyers_std to zero vectors.")
        moment_unmatched_buyers_avg = np.zeros(K_buyers-4)
        moment_unmatched_buyers_std = np.zeros(K_buyers-4)

    # --- Compute Moments for Unmatched Sellers ---
    if unmatched_sellers_predictors:
        x_unmatched_sellers = np.array(unmatched_sellers_predictors)  # Shape: (N_unmatched_sellers, K_sellers)
        x_unmatched_sellers = x_unmatched_sellers[:,:-4]
        avg_unmatched_sellers = np.mean(x_unmatched_sellers, axis=0)  # Shape: (K_sellers,)
        std_unmatched_sellers = np.std(x_unmatched_sellers, axis=0, ddof=0)  # Shape: (K_sellers,)

        moment_unmatched_sellers_avg = avg_unmatched_sellers  # Shape: (K_sellers,)
        moment_unmatched_sellers_std = std_unmatched_sellers  # Shape: (K_sellers,)
    else:
        warnings.warn("No unmatched sellers found. Setting moment_unmatched_sellers_avg and moment_unmatched_sellers_std to zero vectors.")
        moment_unmatched_sellers_avg = np.zeros(K_sellers-4)
        moment_unmatched_sellers_std = np.zeros(K_sellers-4)

    moment_unmatched_buyers = np.stack([moment_unmatched_buyers_avg, moment_unmatched_buyers_std], axis=0)  # Shape: (2, K_buyers, K_buyers)
    moment_unmatched_sellers = np.stack([moment_unmatched_sellers_avg, moment_unmatched_sellers_std], axis=0)  # Shape: (2, K_sellers, K_sellers)

    # Store all moments in a dictionary
    moments = {
        "moment_buyers": moment_buyers,                       # Shape: (2, K_buyers, K_sellers)
        "moment_sellers": moment_sellers,                     # Shape: (2, K_sellers, K_buyers)
        "moment_unmatched_buyers": moment_unmatched_buyers,   # Shape: (2, K_buyers)
        "moment_unmatched_sellers": moment_unmatched_sellers  # Shape: (2, K_sellers)
    }

    return moments

def return_degenerate_moments(K_buyers, K_sellers, M=1e3):
    
    moment_buyers = np.full((2,K_buyers-4,K_sellers-4),M)
    moment_sellers = np.full((2,K_sellers-4,K_buyers-4),M)
    moment_unmatched_buyers = np.full((2,K_buyers-4),M)
    moment_unmatched_sellers = np.full((2,K_sellers-4),M)
    
    # Store all moments in a dictionary
    moments = {
        "moment_buyers": moment_buyers,                       # Shape: (2, K_buyers, K_sellers)
        "moment_sellers": moment_sellers,                     # Shape: (2, K_sellers, K_buyers)
        "moment_unmatched_buyers": moment_unmatched_buyers,   # Shape: (2, K_buyers)
        "moment_unmatched_sellers": moment_unmatched_sellers  # Shape: (2, K_sellers)
    }
    
    return moments

def simulate_and_compute(args, return_simulation = False, compute_matches = False):
    (seed, beta_buyers, beta_sellers, eta, use_kde, method, bandwidth, estimator_params, buyers_threshold_spline, sellers_threshold_spline) = args

    # Unpack estimator parameters
    buyers_data, sellers_data, error_generator_buyers_params, error_generator_sellers_params, K_buyers, K_sellers, h = estimator_params

    buyers_data_sim = buyers_data.copy()
    sellers_data_sim = sellers_data.copy()
    
    
    # Sort the arrays by 'Valuation' in ascending order
    buyers_data_sim.sort(order='ID')
    sellers_data_sim.sort(order='ID')
    
    # Reconstruct error generators
    error_generator_buyers = create_error_generator(*error_generator_buyers_params)
    error_generator_sellers = create_error_generator(*error_generator_sellers_params)

    # Set up random number generator with the seed
    rng = np.random.default_rng(seed)

    # Extract predictors
    X_buyers = buyers_data_sim['Predictors']
    X_sellers = sellers_data_sim['Predictors']

    N_buyers, K_buyers = X_buyers.shape
    N_sellers, K_sellers = X_sellers.shape
    
    # Generate new errors
    errors_buyers = error_generator_buyers.rvs(N_buyers, random_state=rng)
    errors_sellers = error_generator_sellers.rvs(N_sellers, random_state=rng)

    # Compute new valuations
    valuations_buyers = X_buyers @ beta_buyers + errors_buyers
    valuations_sellers = X_sellers @ beta_sellers + errors_sellers

    # Scale valuations
    valuations_buyers /= eta
    valuations_sellers /= eta

    # Add small perturbation only to duplicates
    valuations_buyers = perturb_duplicates(valuations_buyers)
    valuations_sellers = perturb_duplicates(valuations_sellers)
    
    buyers_data_sim['Valuation'] = valuations_buyers
    sellers_data_sim['Valuation'] = valuations_sellers

    # Sort the arrays by 'Valuation' in ascending order
    buyers_data_sim.sort(order='Valuation')
    sellers_data_sim.sort(order='Valuation')

    # Create MarketSide instances
    buyers_side = MarketSide(
        name='Buyers',
        data_array=buyers_data_sim,
        use_kde=use_kde,
        bandwidth=bandwidth,
        method=method
    )
    sellers_side = MarketSide(
        name='Sellers',
        data_array=sellers_data_sim,
        use_kde=use_kde,
        bandwidth=bandwidth,
        method=method
    )

    # Simulate the equilibrium
    simulation = MarketSimulation(buyers_side, sellers_side, h)
    ratios, _ = simulation.check_chi_derivatives()
    ratios_buyers, ratios_sellers = ratios
    if (ratios_buyers < 0.9) | (ratios_sellers < 0.9):
        if (ratios_buyers < 0.9) & (ratios_sellers <0.9):
            print("Too many negative derivatives for buyers and sellers.", flush=True)
            return return_degenerate_moments(K_buyers, K_sellers)
        elif (ratios_buyers < 0.9):
            print("Too many negative derivatives for buyers.", flush=True)
            return return_degenerate_moments(K_buyers, K_sellers)
        else:
            print("Too many negative derivatives for sellers.", flush=True)
            return return_degenerate_moments(K_buyers, K_sellers)
    simulation.evaluate_Delta()
    simulation.compute_omegas()
    simulation.compute_thresholds(buyers_threshold_spline=buyers_threshold_spline, sellers_threshold_spline=sellers_threshold_spline)
    
    if return_simulation:
        if compute_matches:
            simulation.compute_matches()
        return simulation
    
    simulation.compute_matches()
    
    
    # Compute the moments for this simulation
    moments = compute_moments_from_simulation(buyers_data_sim, sellers_data_sim, simulation)

    return moments


class MomentEstimator:
    def __init__(self,
                 observed_data_buyers,
                 observed_data_sellers,
                 simulated_data_params,
                 error_generator_buyers,
                 error_generator_sellers,
                 eta):
        """"
        Initializes the MomentEstimator.

        Parameters:
        - observed_data_buyers: DataFrame with buyer data
        - observed_data_sellers: DataFrame with seller data
        - simulated_data_params: dictionary with simulation parameters
        """
        
        
        
        self.error_generator_buyers = error_generator_buyers
        self.error_generator_sellers = error_generator_sellers
        
        self.eta = eta
        
        self.use_observed_data = observed_data_buyers is not None and observed_data_sellers is not None
        self.buyers_data = observed_data_buyers
        self.sellers_data = observed_data_sellers
        
        self.simulated_data_params = simulated_data_params

        # Dictionaries to store MarketSide instances per market
        self.market_sides = {}
        self.markets_present = True
        self.market_IDs = None
        if self.markets_present:
            self.market_IDs = np.unique(observed_data_buyers['Market'])
            
        self.logger = None
        # Process data accordingly
        self.process_data()
       
    def set_logger(self, name, log_file_path=None, level=logging.INFO):
        """
        Assign a logger to this MomentEstimator instance.
        You can call this before running a particular estimation routine
        to ensure you have a dedicated logger for that run.
        """
        self.logger = get_logger(name, log_file_path, level)
        
    def process_data(self):
        if self.use_observed_data:
            self.process_observed_data()
        else:
            self.generate_data_and_sides()
    
    def process_observed_data(self):
        
        market_IDs = self.market_IDs
        
        if len(market_IDs)>1:
            
            for market_ID in market_IDs:
                
                # Instantiate MarketSide objects
                buyers_side = MarketSide(
                name='Clients',
                data_array=self.buyers_data[self.buyers_data['Market'] == market_ID],
                use_kde=True,        # As per your existing code; adjust if needed
                bandwidth='scott',   # Example bandwidth method; adjust as needed
                method='central'     # Example method; adjust as needed
                )

                sellers_side = MarketSide(
                name='Politicians',
                data_array=self.sellers_data[self.sellers_data['Market'] == market_ID],
                use_kde=True,        # As per your existing code; adjust if needed
                bandwidth='scott',   # Example bandwidth method; adjust as needed
                method='central'     # Example method; adjust as needed
                )
                
                self.market_sides[market_ID] = (buyers_side, sellers_side)
                
        else:
            # Instantiate MarketSide objects
            buyers_side = MarketSide(
            name='Clients',
            data_array=self.buyers_data,
            use_kde=True,        # As per your existing code; adjust if needed
            bandwidth='scott',   # Example bandwidth method; adjust as needed
            method='central'     # Example method; adjust as needed
            )
    
            sellers_side = MarketSide(
            name='Politicians',
            data_array=self.sellers_data,
            use_kde=True,        # As per your existing code; adjust if needed
            bandwidth='scott',   # Example bandwidth method; adjust as needed
            method='central'     # Example method; adjust as needed
            )
        
            # Store in dictionaries
            self.market_sides['single_market'] = (buyers_side, sellers_side)
        
    def generate_data_and_sides(self):
        
        rng = np.random.default_rng(base_seed)
        """
        Generates simulated data and creates MarketSide instances.
        """
        params = self.simulated_data_params
        N_buyers = params['N_buyers']
        N_sellers = params['N_sellers']
        K_buyers = params['K_buyers']
        K_sellers = params['K_sellers']
        beta_true_buyers = params['beta_true_buyers']
        beta_true_sellers = params['beta_true_sellers']
        means_buyers = params['means_buyers']
        means_sellers = params['means_sellers']
        
        
        # Define data types for structured arrays
        buyer_dtype = np.dtype([
            ('ID', np.int64),
            ('Predictors', np.float64, (K_buyers,)),
            ('Valuation', np.float64)
        ])
    
        seller_dtype = np.dtype([
            ('ID', np.int64),
            ('Predictors', np.float64, (K_sellers,)),
            ('Valuation', np.float64)
        ])
    
        # Generate IDs
        buyer_ids = np.arange(N_buyers)
        seller_ids = np.arange(N_sellers)
    
        # Generate predictors
        X_buyers = generate_predictors(
            N=N_buyers,
            K=K_buyers,
            means=means_buyers,
            lower_bounds=means_buyers-2,
            upper_bounds=means_buyers+2,
            scales=np.ones(K_buyers)
        )
    
        # Generate X variables for sellers
        X_sellers = generate_predictors(
            N=N_sellers,
            K=K_sellers,
            means=means_sellers,
            lower_bounds=means_sellers-2,
            upper_bounds=means_sellers+2,
            scales=np.ones(K_sellers)
        )
    
        # Generate errors
        errors_buyers = self.error_generator_buyers.rvs(N_buyers,random_state=rng)
        errors_sellers = self.error_generator_sellers.rvs(N_sellers,random_state=rng)
    
        # Compute valuations
        valuations_buyers = X_buyers @ beta_true_buyers + errors_buyers
        valuations_sellers = X_sellers @ beta_true_sellers + errors_sellers
    
        # Scale valuations
        valuations_buyers /= self.eta
        valuations_sellers /= self.eta
        
        # Add small perturbation only to duplicates
        valuations_buyers = perturb_duplicates(valuations_buyers)
        valuations_sellers = perturb_duplicates(valuations_sellers)
    
        # Create structured arrays
        buyers_array = np.zeros(N_buyers, dtype=buyer_dtype)
        buyers_array['ID'] = buyer_ids
        buyers_array['Predictors'] = X_buyers
        buyers_array['Valuation'] = valuations_buyers
    
        sellers_array = np.zeros(N_sellers, dtype=seller_dtype)
        sellers_array['ID'] = seller_ids
        sellers_array['Predictors'] = X_sellers
        sellers_array['Valuation'] = valuations_sellers
    
        # Sort the arrays by 'Valuation' in ascending order
        buyers_array.sort(order='Valuation')
        sellers_array.sort(order='Valuation')
        # Create MarketSide instances
        buyers_side = MarketSide('Buyers', buyers_array)
        sellers_side = MarketSide('Sellers', sellers_array)
    
        # Store in dictionaries
        self.market_sides['single_market'] = (buyers_side, sellers_side)
        

    def simulate_empirical_equilibrium(self, h):
        """
        Simulate the equilibrium and compute the moments from the generated data.
        
        :param buyers: The MarketSide instance for buyers.
        :param sellers: The MarketSide instance for sellers.
        """
        
        simulations = {}
        for market_key, (buyers_side, sellers_side) in self.market_sides.items():
            # Simulate the equilibrium for this market
            simulation = MarketSimulation(buyers_side, sellers_side, h)
            #simulation.run_equilibrium_simulation()
            simulation.check_chi_derivatives()
            simulation.evaluate_Delta()
            simulation.compute_omegas()
            
            #print("Computing thresholds")
            simulation.compute_thresholds()
            simulation.compute_matches()
            #print("Computing payments")
            #simulation.compute_payments()
            
            # Store the simulation
            simulations[market_key] = simulation
        

        return simulations
    
    
    
        
    # def simulate_equilibrium(self, beta_buyers, beta_sellers, eta, use_kde=True, method='central', bandwidth='scott', seed=None):
    #     """
    #     Simulate the equilibrium for given beta values for buyers and sellers.
    
    #     :param beta_buyers: The beta vector for buyers (array of size K_buyers).
    #     :param beta_sellers: The beta vector for sellers (array of size K_sellers).
    #     :param eta: Scaling parameter for valuations.
    #     :param use_kde: Boolean to specify whether to use KDE for PDF estimation.
    #     :param method: Method for numerical differentiation ('forward', 'backward', 'central').
    #     :param bandwidth: Bandwidth method for KDE if use_kde is True.
    #     :param seed: Seed for random number generation.
    #     :return: The MarketSimulation object after running the equilibrium.
    #     """
    #     # Set the seed if provided
    #     rng = np.random.default_rng(seed)
    
    #     # Extract data arrays
    #     buyers_array = self.buyers_data.copy()
    #     sellers_array = self.sellers_data.copy()
    
    #     # Extract predictors
    #     X_buyers = buyers_array['Predictors']
    #     X_sellers = sellers_array['Predictors']
    
    #     N_buyers = len(buyers_array)
    #     N_sellers = len(sellers_array)
    
    #     # Generate new errors
    #     errors_buyers = self.error_generator_buyers.rvs(N_buyers, random_state=rng)
    #     errors_sellers = self.error_generator_sellers.rvs(N_sellers, random_state=rng)
    
    #     # Compute new valuations with candidate beta values
    #     valuations_buyers = X_buyers @ beta_buyers + errors_buyers
    #     valuations_sellers = X_sellers @ beta_sellers + errors_sellers
    
    #     # Scale valuations
    #     valuations_buyers /= eta
    #     valuations_sellers /= eta
    
    #     # Update valuations in the data arrays
    #     buyers_array['Valuation'] = valuations_buyers
    #     sellers_array['Valuation'] = valuations_sellers
    
    #     # Sort the arrays by 'Valuation' in ascending order
    #     buyers_array.sort(order='Valuation')
    #     sellers_array.sort(order='Valuation')
    
    #     # Create MarketSide instances
    #     buyers_side = MarketSide(
    #         name='Buyers',
    #         data_array=buyers_array,
    #         use_kde=use_kde,
    #         bandwidth=bandwidth,
    #         method=method
    #     )
    #     sellers_side = MarketSide(
    #         name='Sellers',
    #         data_array=sellers_array,
    #         use_kde=use_kde,
    #         bandwidth=bandwidth,
    #         method=method
    #     )
    
    #     # Simulate the equilibrium
    #     simulation = MarketSimulation(buyers_side, sellers_side, self.h)
    #     simulation.check_chi_derivatives()
    #     simulation.evaluate_Delta()
    #     simulation.compute_omegas()
    #     simulation.compute_thresholds()
    #     simulation.compute_matches()
    
    #     # Return the simulation
    #     return simulation

        
    #     return simulation
    
    
    
    # def simulate_and_compute_moments(self, beta_buyers, beta_sellers, use_kde=True, method='central', bandwidth='scott'):
        
    #     """
    #     Runs a single simulation and computes the moments.
    
    #     :param beta_buyers: The beta value for buyers.
    #     :param beta_sellers: The beta value for sellers.
    #     :param use_kde: Boolean to specify whether to use KDE for PDF estimation.
    #     :param method: Method for numerical differentiation ('forward', 'backward', 'central').
    #     :param bandwidth: Bandwidth method for KDE if use_kde is True.
    #     :return: A dictionary containing the computed moments.
    #     """
    #     # Simulate the equilibrium
    #     simulation = self.simulate_equilibrium(beta_buyers, beta_sellers, use_kde, method, bandwidth)
        
    #     # Compute the moments for this simulation
    #     moments = self.compute_moments_from_simulation(simulation)
        
    #     return moments

    def simulate_average_moments(self, beta_buyers, beta_sellers, eta, S=100, use_kde=True, method='central', bandwidth='scott'):
        """
        Simulate the equilibrium S times and compute the average of the moments.
        
        :param beta_buyers: The beta value for buyers.
        :param beta_sellers: The beta value for sellers.
        :param S: Number of simulations to run.
        :param use_kde: Boolean to specify whether to use KDE for PDF estimation.
        :param method: Method for numerical differentiation ('forward', 'backward', 'central').
        :param bandwidth: Bandwidth method for KDE if use_kde is True.
        :return: A dictionary containing the averaged moments.
        """
        
        total_moments = {"moment_buyers": np.zeros((self.K_buyers, self.K_sellers)), "moment_sellers": np.zeros((self.K_sellers, self.K_buyers))}
        
        for s in range(S):
            current_seed = base_seed + s  # Generate a unique seed for each simulation
            
            # Simulate equilibrium (errors generated within the method)
            simulation = self.simulate_equilibrium(beta_buyers, beta_sellers, eta, use_kde, method, bandwidth, seed=current_seed)
            
            # Compute the moments for this simulation
            moments = self.compute_moments_from_simulation(simulation)
            
            # Accumulate the moments
            total_moments["moment_buyers"] += moments["moment_buyers"]
            total_moments["moment_sellers"] += moments["moment_sellers"]
            
            #print(f"Completed simulation {s + 1} out of {S} for beta_b={beta_buyers}, beta_s={beta_sellers}")
         
        # Average the moments over the S simulations
        average_moments = {key: total_moments[key] / S for key in total_moments}
        
        return average_moments

   
    def simulate_average_moments_parallelized(self, beta_buyers, beta_sellers, eta, h, S, use_kde=True, method='central', bandwidth='scott', seeds=None, two_step_thresholds=True):
    
        """
        Simulate the market S times in parallel and compute the average moments.
    
        Parameters:
        - beta_buyers (np.ndarray): Parameter vector for buyers.
        - beta_sellers (np.ndarray): Parameter vector for sellers.
        - eta (float): Scaling parameter.
        - h (float): Additional simulation parameter.
        - S (int): Number of simulations.
        - use_kde (bool): Whether to use KDE in MarketSide.
        - method (str): Method for KDE or other processing.
        - bandwidth (str): Bandwidth parameter for KDE.
        - seeds (np.ndarray or None): Array of seeds for simulations.
        - return_moments_list (bool): Whether to return the list of all moments.
    
        Returns:
        - dict: Averaged moments across all simulations.
        - list (optional): List of moments from each simulation if return_moments_list is True.
        """
        # ------------------------------
        # 1. Prepare the necessary parameters to pass
        # ------------------------------
        # Extract error generator parameters
        error_generator_buyers_params = (
            self.error_generator_buyers.dist.name,
            self.error_generator_buyers.args
        )
        error_generator_sellers_params = (
            self.error_generator_sellers.dist.name,
            self.error_generator_sellers.args
        )
    
        average_moments = {}
        
        IDs_list=list(self.market_IDs)
        for market_ID in IDs_list:
            print(market_ID)
            # Retrieve buyer and seller data
            buyers = self.market_sides[market_ID][0]
            sellers = self.market_sides[market_ID][1]
        
            # Number of predictors for buyers and sellers
            K_buyers = buyers.data['Predictors'].shape[1]
            K_sellers = sellers.data['Predictors'].shape[1]
        
            # Bundle estimator parameters
            estimator_params = (
                buyers.data,
                sellers.data,
                error_generator_buyers_params,
                error_generator_sellers_params,
                K_buyers,
                K_sellers,
                h  # Include h
            )
                    # ------------------------------
            # 2. Generate or assign seeds
            # ------------------------------
            if seeds is None:
                seeds = np.arange(S+1) + base_seed  # Ensure base_seed is defined within the class
            args_list = [
                (
                    seed,
                    beta_buyers,
                    beta_sellers,
                    eta,
                    use_kde,
                    method,
                    bandwidth,
                    estimator_params,
                    None,
                    None
                )
                for seed in seeds
            ]
            
            if two_step_thresholds:
                # Run the first simulation without computing matches
                try:
                    first_simulation = simulate_and_compute(args_list[0], return_simulation=True)
                    if not isinstance(first_simulation, MarketSimulation):
                        average_moments[market_ID] = first_simulation
                        break
                except Exception as e:
                    average_moments[market_ID] = return_degenerate_moments(K_buyers, K_sellers)
                    break
                    
                # Generate splines from the first simulation
                buyers_threshold_spline, sellers_threshold_spline = first_simulation.generate_threshold_splines()
                
                # Prepare arguments for the remaining S simulations with splines
                modified_args_list = []
                for arg in args_list[1:S + 1]:
                    modified_args = (
                        arg[0],  # seed
                        arg[1],  # beta_buyers
                        arg[2],  # beta_sellers
                        arg[3],  # eta
                        arg[4],  # use_kde
                        arg[5],  # method
                        arg[6],  # bandwidth
                        arg[7],  # estimator_params
                        buyers_threshold_spline,  # Spline for buyers
                        sellers_threshold_spline   # Spline for sellers
                    )
                    modified_args_list.append(modified_args)
            else:
                # If not using two-step thresholds, prepare all S + 1 simulations normally
                modified_args_list = args_list[:S]
            
            num_simulations = len(modified_args_list)
    
            # ------------------------------
            # 3. Initialize moment sums and counters
            # ------------------------------
            # Matched Moments
            total_moment_buyers = np.zeros((2,K_buyers-4, K_sellers-4))
            total_moment_sellers = np.zeros((2,K_sellers-4, K_buyers-4))
        
            # Unmatched Moments
            total_moment_unmatched_buyers = np.zeros((2,K_buyers-4))
            total_moment_unmatched_sellers = np.zeros((2,K_sellers-4))
        
            # Counter for successful simulations
            successful_simulations = 0
        
            # ------------------------------
            # 4. Submit simulations and handle exceptions
            
            # Get the number of available CPUs (should match Slurm allocations)
            # Retrieve the number of allocated CPUs
            
            #print(f"Available CPUs for this job: {available_cpus}", flush=True)
            
            
            # ------------------------------
            
            for args in modified_args_list:
                
                try:
                    moments = simulate_and_compute(args)
                    # Aggregate matched moments
                    total_moment_buyers += moments['moment_buyers']
                    total_moment_sellers += moments['moment_sellers']
    
                    # Aggregate unmatched moments
                    total_moment_unmatched_buyers += moments['moment_unmatched_buyers']
                    total_moment_unmatched_sellers += moments['moment_unmatched_sellers']
    
                    successful_simulations += 1
                except Exception as e:
                    print(f"A simulation failed with exception: {e}", flush=True)
                    # Optionally, log the exception details
                    break  # Skip to the next future
    
            
    # =============================================================================
    #         with ProcessPoolExecutor(max_workers=available_cpus) as executor:
    #             # Submit each simulation as a future
    #             futures = {executor.submit(simulate_and_compute, args): args for args in modified_args_list}
    #     
    #             for future in as_completed(futures):
    #                 args = futures[future]
    #                 try:
    #                     moments = future.result()
    #                     # Aggregate matched moments
    #                     total_moment_buyers += moments['moment_buyers']
    #                     total_moment_sellers += moments['moment_sellers']
    #     
    #                     # Aggregate unmatched moments
    #                     total_moment_unmatched_buyers += moments['moment_unmatched_buyers']
    #                     total_moment_unmatched_sellers += moments['moment_unmatched_sellers']
    #     
    #                     successful_simulations += 1
    #                 except Exception as e:
    #                     print(f"A simulation failed with exception: {e}", flush=True)
    #                     # Optionally, log the exception details
    #                     continue  # Skip to the next future
    # =============================================================================
        
            # ------------------------------
            # 5. Check if there are any successful simulations
            # ------------------------------
            if successful_simulations == 0:
                average_moments_market = return_degenerate_moments(K_buyers, K_sellers)
                #raise ValueError("All simulations failed. Cannot compute average moments.")
        
            # ------------------------------
            # 6. Compute average moments over successful simulations
            # ------------------------------
            average_moments_market = {
                'moment_buyers': total_moment_buyers / successful_simulations,                         # Shape: (2, K_buyers, K_sellers)
                'moment_sellers': total_moment_sellers / successful_simulations,                       # Shape: (2, K_sellers, K_buyers)
                'moment_unmatched_buyers': total_moment_unmatched_buyers / successful_simulations,     # Shape: (2, K_buyers)
                'moment_unmatched_sellers': total_moment_unmatched_sellers / successful_simulations    # Shape: (2, K_sellers)
            }
            
            average_moments[market_ID] = average_moments_market
        # ------------------------------
        # 7. Return results
        # ------------------------------
        return average_moments

    def compute_moment_differences(self, true_moments, simulated_moments):
        """
        Compute the differences between true and simulated moments.
    
        :param true_moments: Dictionary containing the true moments.
        :param simulated_moments: Dictionary containing the simulated moments.
        :return: A dictionary containing the differences between true and simulated moments.
        """
        moment_buyers_diff = true_moments["moment_buyers"] - simulated_moments["moment_buyers"]
        moment_sellers_diff = true_moments["moment_sellers"] - simulated_moments["moment_sellers"]
        
        moment_diff = {
            "moment_buyers_diff": moment_buyers_diff,
            "moment_sellers_diff": moment_sellers_diff
        }
        
        return moment_diff

    def estimate_betas(self, beta_initial, true_moments, K_buyers, K_sellers, eta, h, S=10, W = None, method='L-BFGS-B', bounds=None, options=None, log_file=None):
        """
        Estimate beta values by minimizing the objective function using the provided gradient.
        """
        # First Stage: Use identity matrix
        #W_identity = np.eye(K_buyers * K_sellers)
        
        # Generate seeds once to use in all simulations
        seeds = np.arange(S) + base_seed

        # Define the objective and gradient functions
        objective = lambda beta: self.objective_function(
            beta, true_moments, K_buyers, K_sellers, eta, h, S, W, seeds=seeds
        )
        gradient = lambda beta: self.gradient_objective(
            beta, true_moments, K_buyers, K_sellers, eta, h, S, W, seeds=seeds
        )
        
        # Run the optimization
        result = minimize(
            objective,
            beta_initial,
            method=method,
            #jac=gradient,
            bounds=bounds,
            options=options
        )
        
        return result
        
    def distance_between_moments(self, moments_true, moments_sim):
        """
        Compute the distance between the true moments and simulated moments.
        
        :param moments_true: Dictionary containing the true moments.
        :param moments_sim: Dictionary containing the simulated moments.
        :return: The Euclidean distance between the true and simulated moments.
        """
        moment_buyers_diff = moments_true["moment_buyers"] - moments_sim["moment_buyers"]
        moment_sellers_diff = moments_true["moment_sellers"] - moments_sim["moment_sellers"]
        return np.sqrt(moment_buyers_diff**2 + moment_sellers_diff**2)
    
    def compute_optimal_weighting_matrix(self, beta, true_moments_all, K_buyers, K_sellers, eta, h, S, W, use_kde=True, method='central', bandwidth='scott', seeds=None):
        """
        Objective function to minimize during estimation.
    
        This function computes the weighted squared differences between
        simulated moments and true moments, considering both average and std deviations.
    
        Parameters:
        - beta (np.ndarray): Parameter vector [beta_buyers | beta_sellers].
        - true_moments (dict): Dictionary containing true moments with keys:
            - 'moment_buyers': np.ndarray of shape (2, K_buyers, K_sellers)
            - 'moment_sellers': np.ndarray of shape (2, K_sellers, K_buyers)
        - K_buyers (int): Number of buyer predictors.
        - K_sellers (int): Number of seller predictors.
        - eta, h, S: Additional simulation parameters.
        - W (np.ndarray): Weighting matrix for the moments.
        - use_kde, method, bandwidth, seeds: Additional parameters for simulation.
    
        Returns:
        - objective_value (float): The computed objective function value.
        """
        
        print(f"Beta vector: {beta}", flush=True)
        # ------------------------------
        # 1. Extract beta_buyers and beta_sellers
        # ------------------------------
        beta_buyers = beta[:K_buyers]                          # Shape: (K_buyers,)
        beta_sellers = beta[K_buyers:K_buyers + K_sellers]    # Shape: (K_sellers,)
    
        # ------------------------------
        # 2. Simulate moments using the candidate beta values
        # ------------------------------
        simulated_moments_all = self.simulate_average_moments_parallelized(
            beta_buyers, beta_sellers, eta, h, S, use_kde, method, bandwidth, seeds=seeds, two_step_thresholds=True
        )
        
        # ------------------------------
        # 3. Stack the true moments for buyers, sellers, unmatched buyers, and unmatched sellers
        # ------------------------------
        # Shape: (2, K_buyers, K_sellers)
        # Initialize an accumulator for sums
        true_moments = {}
        simulated_moments = {}
        # Loop over each year in the dictionary
        
        M_matrix = np.zeros(W.shape)
        for year in simulated_moments_all.keys():
            
            true_moments = true_moments_all[year]
            true_moment_buyers = true_moments['moment_buyers']  
            # Shape: (2, K_sellers, K_buyers)
            true_moment_sellers = true_moments['moment_sellers']
            # Shape: (2, K_buyers)
            true_moment_unmatched_buyers = true_moments['moment_unmatched_buyers']
            # Shape: (2, K_sellers)
            true_moment_unmatched_sellers = true_moments['moment_unmatched_sellers']
            
            
            simulated_moments = simulated_moments_all[year]
            # Shape: (2, K_buyers, K_sellers)
            simulated_moment_buyers = simulated_moments['moment_buyers']  
            
            # Shape: (2, K_sellers, K_buyers)
            simulated_moment_sellers = simulated_moments['moment_sellers']  
            
            # Shape: (2, K_buyers)
            simulated_moment_unmatched_buyers = simulated_moments['moment_unmatched_buyers']  
            
            # Shape: (2, K_sellers)
            simulated_moment_unmatched_sellers = simulated_moments['moment_unmatched_sellers']  
            
                    
            # ------------------------------
            # 5. Compute the differences between true and simulated moments
            # ------------------------------
            # Buyers' Moment Differences: Shape (2, K_buyers, K_sellers)
            moment_diff_buyers = true_moment_buyers - simulated_moment_buyers  
            
            # Sellers' Moment Differences: Shape (2, K_sellers, K_buyers)
            moment_diff_sellers = true_moment_sellers - simulated_moment_sellers  
            
            # Unmatched Buyers' Moment Differences: Shape (2, K_buyers)
            moment_diff_unmatched_buyers = true_moment_unmatched_buyers - simulated_moment_unmatched_buyers  
            
            # Unmatched Sellers' Moment Differences: Shape (2, K_sellers)
            moment_diff_unmatched_sellers = true_moment_unmatched_sellers - simulated_moment_unmatched_sellers  
                
                
            # ------------------------------
            # 6. Flatten the moment differences into vectors
            # ------------------------------
            # Buyers' Moments
            # Shape before flatten: (2, K_buyers, K_sellers)
            # Shape after flatten: (2 * K_buyers * K_sellers,)
            moment_diff_buyers_flat = moment_diff_buyers.flatten()  
            
            # Sellers' Moments
            # Shape before flatten: (2, K_sellers, K_buyers)
            # Shape after flatten: (2 * K_sellers * K_buyers,)
            moment_diff_sellers_flat = moment_diff_sellers.flatten()  
            
            # Unmatched Buyers' Moments
            # Shape before flatten: (2, K_buyers)
            # Shape after flatten: (2 * K_buyers,)
            moment_diff_unmatched_buyers_flat = moment_diff_unmatched_buyers.flatten()
            
            # Unmatched Sellers' Moments
            # Shape before flatten: (2, K_sellers)
            # Shape after flatten: (2 * K_sellers,)
            moment_diff_unmatched_sellers_flat = moment_diff_unmatched_sellers.flatten()
            
                
            # ------------------------------
            # 7. Concatenate all moment differences into a single vector
            #    Ordering: [buyers_avg, buyers_std, sellers_avg, sellers_std, unmatched_buyers_avg, unmatched_buyers_std, unmatched_sellers_avg, unmatched_sellers_std]
            # ------------------------------
            M = np.concatenate([
                moment_diff_buyers_flat*np.sqrt(K_sellers),
                moment_diff_sellers_flat*np.sqrt(K_buyers),
                moment_diff_unmatched_buyers_flat*np.sqrt(K_sellers),
                moment_diff_unmatched_sellers_flat*np.sqrt(K_buyers)
            ])  # Shape: (4 * K_buyers * K_sellers + 2 * K_buyers + 2 * K_sellers,)
            
            M_matrix += np.outer(M, M.T)
            
        return M_matrix/len(simulated_moments_all.keys())
    
    
    def objective_function(self, beta, true_moments_all, K_buyers, K_sellers, eta, h, S, W, use_kde=True, method='central', bandwidth='scott', seeds=None):
        """
        Objective function to minimize during estimation.
    
        This function computes the weighted squared differences between
        simulated moments and true moments, considering both average and std deviations.
    
        Parameters:
        - beta (np.ndarray): Parameter vector [beta_buyers | beta_sellers].
        - true_moments (dict): Dictionary containing true moments with keys:
            - 'moment_buyers': np.ndarray of shape (2, K_buyers, K_sellers)
            - 'moment_sellers': np.ndarray of shape (2, K_sellers, K_buyers)
        - K_buyers (int): Number of buyer predictors.
        - K_sellers (int): Number of seller predictors.
        - eta, h, S: Additional simulation parameters.
        - W (np.ndarray): Weighting matrix for the moments.
        - use_kde, method, bandwidth, seeds: Additional parameters for simulation.
    
        Returns:
        - objective_value (float): The computed objective function value.
        """
        
        print(f"Beta vector: {beta}", flush=True)
        # ------------------------------
        # 1. Extract beta_buyers and beta_sellers
        # ------------------------------
        beta_buyers = beta[:K_buyers]                          # Shape: (K_buyers,)
        beta_sellers = beta[K_buyers:K_buyers + K_sellers]    # Shape: (K_sellers,)
    
        # ------------------------------
        # 2. Simulate moments using the candidate beta values
        # ------------------------------
        simulated_moments_all = self.simulate_average_moments_parallelized(
            beta_buyers, beta_sellers, eta, h, S, use_kde, method, bandwidth, seeds=seeds, two_step_thresholds=True
        )
        
        # ------------------------------
        # 3. Stack the true moments for buyers, sellers, unmatched buyers, and unmatched sellers
        # ------------------------------
        # Shape: (2, K_buyers, K_sellers)
        # Initialize an accumulator for sums
        true_moments = {}
        simulated_moments = {}
        # Loop over each year in the dictionary
        
        final_objective = 0
        for year in simulated_moments_all.keys():
            
            true_moments = true_moments_all[year]
            true_moment_buyers = true_moments['moment_buyers']  
            # Shape: (2, K_sellers, K_buyers)
            true_moment_sellers = true_moments['moment_sellers']
            # Shape: (2, K_buyers)
            true_moment_unmatched_buyers = true_moments['moment_unmatched_buyers']
            # Shape: (2, K_sellers)
            true_moment_unmatched_sellers = true_moments['moment_unmatched_sellers']
            
            
            simulated_moments = simulated_moments_all[year]
            # Shape: (2, K_buyers, K_sellers)
            simulated_moment_buyers = simulated_moments['moment_buyers']  
            
            # Shape: (2, K_sellers, K_buyers)
            simulated_moment_sellers = simulated_moments['moment_sellers']  
            
            # Shape: (2, K_buyers)
            simulated_moment_unmatched_buyers = simulated_moments['moment_unmatched_buyers']  
            
            # Shape: (2, K_sellers)
            simulated_moment_unmatched_sellers = simulated_moments['moment_unmatched_sellers']  
            
                    
            # ------------------------------
            # 5. Compute the differences between true and simulated moments
            # ------------------------------
            # Buyers' Moment Differences: Shape (2, K_buyers, K_sellers)
            moment_diff_buyers = true_moment_buyers - simulated_moment_buyers  
            
            # Sellers' Moment Differences: Shape (2, K_sellers, K_buyers)
            moment_diff_sellers = true_moment_sellers - simulated_moment_sellers  
            
            # Unmatched Buyers' Moment Differences: Shape (2, K_buyers)
            moment_diff_unmatched_buyers = true_moment_unmatched_buyers - simulated_moment_unmatched_buyers  
            
            # Unmatched Sellers' Moment Differences: Shape (2, K_sellers)
            moment_diff_unmatched_sellers = true_moment_unmatched_sellers - simulated_moment_unmatched_sellers  
                
                
            # ------------------------------
            # 6. Flatten the moment differences into vectors
            # ------------------------------
            # Buyers' Moments
            # Shape before flatten: (2, K_buyers, K_sellers)
            # Shape after flatten: (2 * K_buyers * K_sellers,)
            moment_diff_buyers_flat = moment_diff_buyers.flatten()  
            
            # Sellers' Moments
            # Shape before flatten: (2, K_sellers, K_buyers)
            # Shape after flatten: (2 * K_sellers * K_buyers,)
            moment_diff_sellers_flat = moment_diff_sellers.flatten()  
            
            # Unmatched Buyers' Moments
            # Shape before flatten: (2, K_buyers)
            # Shape after flatten: (2 * K_buyers,)
            moment_diff_unmatched_buyers_flat = moment_diff_unmatched_buyers.flatten()
            
            # Unmatched Sellers' Moments
            # Shape before flatten: (2, K_sellers)
            # Shape after flatten: (2 * K_sellers,)
            moment_diff_unmatched_sellers_flat = moment_diff_unmatched_sellers.flatten()
            
                
            # ------------------------------
            # 7. Concatenate all moment differences into a single vector
            #    Ordering: [buyers_avg, buyers_std, sellers_avg, sellers_std, unmatched_buyers_avg, unmatched_buyers_std, unmatched_sellers_avg, unmatched_sellers_std]
            # ------------------------------
            M = np.concatenate([
                moment_diff_buyers_flat*np.sqrt(K_sellers),
                moment_diff_sellers_flat*np.sqrt(K_buyers),
                moment_diff_unmatched_buyers_flat*np.sqrt(K_sellers),
                moment_diff_unmatched_sellers_flat*np.sqrt(K_buyers)
            ])  # Shape: (4 * K_buyers * K_sellers + 2 * K_buyers + 2 * K_sellers,)
    
            # ------------------------------
            # 8. Compute the objective function value
            #    objective_value = M.T @ W @ M
            #print(M.shape)
            #print(W.shape)
            # ------------------------------
            
            #print(M)
            objective_value = M.T @ W @ M
            
            final_objective += objective_value
            
        objective_value = final_objective/len(simulated_moments_all.keys())
        
        # Optionally, print the objective value for debugging
        print(f"Objective Value: {objective_value}", flush=True)
        
        return objective_value
    
    def gradient_objective(self, beta, true_moments, K_buyers, K_sellers, eta, h, S, W, use_kde=True, method='central', bandwidth='scott', delta_beta=1e-1, seeds=None):
      """
      Compute the gradient of the objective function with respect to beta using finite differences.
      """
      # Extract beta_buyers and beta_sellers
      beta_buyers = beta[:K_buyers]
      beta_sellers = beta[K_buyers:K_buyers + K_sellers]
      
      # Compute m_sim(beta)
      m_sim_beta = self.simulate_average_moments_parallelized(
          beta_buyers, beta_sellers, eta, h, S, use_kde, method, bandwidth, seeds=seeds
      )
      
      # Compute m_diff = m_true - m_sim_beta
      moment_true_avg = 0.5 * (true_moments['moment_buyers'] + true_moments['moment_sellers'].T)
      moment_simulated_avg = 0.5 * (m_sim_beta['moment_buyers'] + m_sim_beta['moment_sellers'].T)
      m_diff = moment_true_avg - moment_simulated_avg
      M = m_diff.flatten()
      
      # Initialize gradient vector
      grad = np.zeros_like(beta)
      
      # Loop over each parameter in beta
      for i in range(len(beta)):
          beta_perturbed = beta.copy()
          beta_perturbed[i] += delta_beta*beta_perturbed[i]  # Perturb beta_i
          beta_buyers_perturbed = beta_perturbed[:K_buyers]
          beta_sellers_perturbed = beta_perturbed[K_buyers:K_buyers + K_sellers]
          
          # Compute m_sim(beta_perturbed)
          m_sim_beta_perturbed = self.simulate_average_moments_parallelized(
              beta_buyers_perturbed, beta_sellers_perturbed, eta, h, S, use_kde, method, bandwidth, seeds=seeds
          )
          
          # Compute m_diff_perturbed = m_true - m_sim_beta_perturbed
          moment_simulated_avg_perturbed = 0.5 * (m_sim_beta_perturbed['moment_buyers'] + m_sim_beta_perturbed['moment_sellers'].T)
          m_diff_perturbed = moment_true_avg - moment_simulated_avg_perturbed
          M_perturbed = m_diff_perturbed.flatten()
          
          # Approximate derivative: dM_dbeta_i ≈ (M_perturbed - M) / delta_beta
          dM_dbeta_i = (M_perturbed - M) / delta_beta
          
          # Compute gradient component: grad_i = -2 * (dM_dbeta_i^T) * W * M
          grad[i] = -2 * dM_dbeta_i.T @ W @ M
      
      return grad
  
    def estimate_betas_local(self, beta_initial, true_moments, K_buyers, K_sellers, eta, h, S=10, W=None, seeds=None, method='L-BFGS-B', bounds=None, options=None):
        """
        Perform local optimization to estimate beta parameters.
        
        Parameters:
        - beta_initial: Initial guess for beta parameters.
        - true_moments: Dictionary containing the true moments.
        - K_buyers: Number of buyer predictors.
        - K_sellers: Number of seller predictors.
        - eta: Scaling parameter.
        - h: Additional simulation parameter.
        - S: Number of simulations.
        - W: Weighting matrix for the moments. If None, use identity matrix.
        - seeds: Optional seeds for simulations.
        - method: Optimization method (e.g., 'L-BFGS-B').
        - bounds: Bounds for beta parameters.
        - options: Dictionary of solver options.
        
        Returns:
        - result: OptimizationResult object containing the optimization outcome.
        """
        
        
        if W is None:
            # Use identity matrix if no weighting matrix is provided
            moment_vector_length = 2 * (K_buyers * K_sellers + K_sellers * K_buyers + K_buyers + K_sellers)
            W = np.eye(moment_vector_length)
        
        # Define the objective function for the optimizer
        def objective(beta):
            return self.objective_function(
                beta, true_moments, K_buyers, K_sellers, eta, h, S, W, seeds=seeds
            )
        
        def log_callback(xk):
            obj_val = objective(xk)
            print(f"Iteration: parameters={xk}, objective={obj_val}")
            if self.logger is not None:
                self.logger.info(f"Iteration: parameters={xk}, objective={obj_val}")
            return False
        
        # Run the local optimization
        result = minimize(
            objective,
            beta_initial,
            method=method,
            bounds=bounds,
            callback=log_callback,
            options=options
        )
        
        return result
    
    def estimate_betas_global(self, beta_bounds, true_moments, K_buyers, K_sellers, eta, h, S=10, W=None, seeds=None, workers=5):
        """
        Perform global optimization to estimate beta parameters.
        
        Parameters:
        - beta_bounds: List of tuples specifying the bounds for each beta parameter.
        - true_moments: Dictionary containing the true moments.
        - K_buyers: Number of buyer predictors.
        - K_sellers: Number of seller predictors.
        - eta: Scaling parameter.
        - h: Additional simulation parameter.
        - S: Number of simulations.
        - W: Weighting matrix for the moments. If None, use identity matrix.
        - seeds: Optional seeds for simulations.
        
        Returns:
        - result: OptimizationResult object containing the optimization outcome.
        """
        
            
        if W is None:
            # Use identity matrix if no weighting matrix is provided
            moment_vector_length = 2 * (2*(K_buyers-1) * (K_sellers-1) + K_buyers + K_sellers)
            W = np.eye(moment_vector_length)
        
        # Define the objective function for the optimizer
        def objective(beta):
            return self.objective_function(
                beta, true_moments, K_buyers, K_sellers, eta, h, S, W, seeds=seeds
            )
      
        def log_callback(x, convergence):
            obj_val = objective(x)
            print(f"Iteration: parameters={x}, objective={obj_val}, convergence={convergence}")
            if self.logger is not None:
                print("Callback:")  # Debug print
                self.logger.info(f"Iteration: parameters={x}, objective={obj_val}, convergence={convergence}")
            return False
      
        # Run the global optimization using Differential Evolution
        result = differential_evolution(
            objective,
            bounds=beta_bounds,
            strategy='best1bin',
            maxiter=20,
            popsize=5,
            updating='immediate',
            tol=1e-2,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            callback=log_callback,
            disp=True,
            polish=True,
            init='latinhypercube'
        )
        
        return result