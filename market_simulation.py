#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:49:39 2024

@author: michelev
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import CubicSpline
from statsmodels.distributions.empirical_distribution import ECDF
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
import time
from Base import empirical_pdf
from Base import compute_omega, compute_threshold, compute_thresholds_vector, compute_thresholds_vector_min, compute_linear_guess, compute_quadratic_guess, compute_cubic_spline_guess
from Base import euler_equation_long, compute_payments_from_envelope, compute_payment_lower, chi, compute_and_check_increasing

#%%
class MarketSide:
    
    def __init__(self, name, data_array, use_kde=True, bandwidth='scott', method='central'):
        
        self.name = name
        self.N = len(data_array)
        self.data = data_array
        
        self.use_kde = use_kde
        self.bandwidth = bandwidth
        self.method = method  # Method for numerical differentiation
        
        
        # Calculate empirical CDF for buyers and sellers
        
        if not np.all(data_array['Valuation'] == 0):
            
            
            self.valuations = data_array['Valuation']
            self.v_upper = data_array['Valuation'].max()
            self.v_lower = data_array['Valuation'].min()
            ecdf = ECDF(self.valuations)
            
            self.distribution = ecdf
            
            # Initialize f_values and interpolator if using numerical differentiation
            if isinstance(self.distribution, ECDF):
                # Filter out -inf and inf from ecdf.x
                v_values = self.distribution.x[np.isfinite(self.distribution.x)]
                F_values = self.distribution.y[np.isfinite(self.distribution.x)]
                if use_kde:
                    kde = gaussian_kde(v_values, bw_method=self.bandwidth)
                    self.f_function = kde.evaluate  # KDE-based PDF
                else:
                    f_values = empirical_pdf(v_values, F_values, method=self.method)
                    self.f_function = CubicSpline(v_values, f_values, bc_type='clamped')  # Spline-based PDF
                self.v_values = v_values  # Store v_values for interpolation
            else:
                self.f_function = None
                self.f_values = None
                self.v_values = None

    def F(self, v):
        if hasattr(self.distribution, 'cdf'):
            return self.distribution.cdf(v)
        elif isinstance(self.distribution, ECDF):
            return self.distribution(v)
        else:
            return self.distribution(v)

    def f(self, v):
        if self.f_function is not None:
            f_value = self.f_function(v)
            
            if isinstance(f_value, np.ndarray) and f_value.size == 1:
                return f_value.item()  # Extract the scalar from the array
            return f_value
        elif hasattr(self.distribution, 'pdf'):
            return self.distribution.pdf(v)
        else:
            h = 1e-5
            return (self.F(v + h) - self.F(v - h)) / (2 * h)
        

    def get_valuations(self):
        return self.valuations

    def __repr__(self):
        return f"{self.name}: {self.N} agents"

class MarketSimulation:
    
    def __init__(self, buyers, sellers, h):
        self.buyers = buyers
        self.sellers = sellers
        self.h = h
        self.buyers_thresholds = None
        self.sellers_thresholds = None
        self.buyers_omega = None
        self.sellers_omega = None
        self.buyers_payments = None
        self.sellers_payments = None
        self.theoretical_valuations_buyers = None
        self.theoretical_valuations_sellers = None
        self.theoretical_thresholds_buyers = None
        self.theoretical_thresholds_sellers = None
        self.theoretical_payments_buyers = None
        self.theoretical_payments_sellers = None
        self.buyers_matches = None
        self.sellers_matches = None
        self.separation = None
        self.bunching_buyers = None
        self.bunching_sellers = None
        self.exclusion_buyers = None
        self.exclusion_sellers = None
        self.buyers_separating_range = None
        self.sellers_separating_range = None
        
    def get_valuations_and_thresholds(self):
        
        buyers_valuations = self.buyers.valuations
        buyers_thresholds = self.buyers_thresholds
        
        sellers_valuations = self.sellers.valuations
        sellers_thresholds = self.sellers_thresholds
        
        return (buyers_valuations, buyers_thresholds), (sellers_valuations, sellers_thresholds)
    
        
    def check_chi_derivatives(self):
         """
         Check the numeric derivatives of chi for both buyers and sellers.
         Prints the number of positive derivatives out of the total.
         """
         
         
         chi_values_buyers = [chi(v, self.buyers.F, self.buyers.f, self.buyers.v_upper, self.h) for v in  self.buyers.valuations]
         chi_values_sellers = [chi(v, self.sellers.F, self.sellers.f, self.sellers.v_upper, self.h) for v in self.sellers.valuations]
         
         num_positive_buyers, total_buyers = compute_and_check_increasing(self.buyers.valuations, chi_values_buyers)
         num_positive_sellers, total_sellers = compute_and_check_increasing(self.sellers.valuations, chi_values_sellers)
            
         print(f"Buyers: {num_positive_buyers} positive derivatives out of {total_buyers}", flush=True)
         print(f"Sellers: {num_positive_sellers} positive derivatives out of {total_sellers}", flush=True)
     
        
         return (num_positive_buyers/total_buyers,  num_positive_sellers/total_sellers), (chi_values_buyers, chi_values_sellers)
         
    def evaluate_Delta(self):
        Delta = euler_equation_long(self.buyers.v_lower, self.sellers.v_lower, self.buyers.F, self.buyers.f, self.sellers.F, self.sellers.f, self.buyers.v_upper, self.sellers.v_upper, self.h)
        print(f"Delta(v_lower_k, v_lower_l) is  {Delta}", flush=True)
        if Delta >= 0:
            self.separation = False
            return "No Separation"
        else:
            self.separation = True
            Delta_k = euler_equation_long(self.buyers.v_upper, self.sellers.v_lower, self.buyers.F, self.buyers.f, self.sellers.F, self.sellers.f, self.buyers.v_upper, self.sellers.v_upper, self.h)
            #Delta_l = euler_equation(self.sellers.v_upper, self.buyers.v_lower, self.buyers.F, self.buyers.f, self.sellers.F, self.sellers.f, self.buyers.v_upper, self.sellers.v_upper, self.h)
            print(f"Delta(v_upper_k, v_lower_l) is  {Delta_k}", flush=True)
            if Delta_k > 0:
                
                self.bunching_buyers = True
                self.exclusion_sellers = False
                
                self.exclusion_buyers = False
                self.bunching_sellers = True
                return "Bunching at the Top"
            elif Delta_k < 0:
                self.exclusion_buyers = True
                self.bunching_sellers = False
                
                self.bunching_buyers = False
                self.exclusion_sellers = True
                return "Exclusion at the Bottom"
            else:
                return "Ambiguous Case"
           
    def compute_omegas(self):
        self.buyers_omega = compute_omega(self.buyers.F, self.buyers.f, self.sellers.F, self.sellers.f, self.buyers.v_upper, self.buyers.v_lower, self.sellers.v_upper, self.h)
        self.sellers_omega = compute_omega(self.sellers.F, self.sellers.f, self.buyers.F, self.buyers.f, self.sellers.v_upper, self.sellers.v_lower, self.buyers.v_upper, self.h)
        
        self.buyers_threshold_omega = compute_threshold(self.sellers_omega, self.sellers.F, self.sellers.f, self.buyers.F, self.buyers.f, self.sellers.v_upper, self.sellers.v_lower, self.buyers.v_upper, self.buyers.v_lower, self.h, (self.sellers.v_lower+self.sellers.v_upper)/2)
        self.sellers_threshold_omega = compute_threshold(self.buyers_omega, self.buyers.F, self.buyers.f, self.sellers.F, self.sellers.f, self.buyers.v_upper, self.buyers.v_lower, self.sellers.v_upper, self.sellers.v_lower, self.h, (self.buyers.v_lower+self.buyers.v_upper)/2)

        # Define separating ranges based on the conditions of exclusion and bunching
        self.buyers_separating_range = [
            self.buyers_omega if self.exclusion_buyers else self.buyers.v_lower,
            self.buyers_threshold_omega if self.bunching_buyers else self.buyers.v_upper
        ]
        self.sellers_separating_range = [
            self.sellers_omega if self.exclusion_sellers else self.sellers.v_lower,
            self.sellers_threshold_omega if self.bunching_sellers else self.sellers.v_upper
        ]

    def compute_thresholds_for_values(self, valuations, side_k, side_l, separating_range_k, separating_range_l, coeffs=None):
        thresholds = []
        i = 1
        for v in valuations:
            #print(i)
            if v <= min(separating_range_k):
                thresholds.append(side_l.v_upper)
            elif v >= max(separating_range_k):
                thresholds.append(side_l.v_lower)
            else:
                if coeffs is not None:
                    if isinstance(coeffs, CubicSpline):
                        # If coeffs is a CubicSpline, evaluate it directly
                        t_guess = coeffs(v)
                    elif isinstance(coeffs, np.ndarray):
                        # If coeffs is an array of coefficients (quadratic), use np.polyval
                        t_guess = np.polyval(coeffs, v)
                    else:
                        raise ValueError("Unrecognized coefficient type. Must be CubicSpline or array of length 3 for quadratic.")
                else:
                # If no coefficients are provided, use the midpoint as the initial guess
                    t_guess = (side_l.v_lower + side_l.v_upper) / 2
                
                # Compute the threshold using the educated guess
                thresholds.append(compute_threshold(
                    v, 
                    side_k.F, 
                    side_k.f, 
                    side_l.F, 
                    side_l.f, 
                    side_k.v_upper, 
                    side_k.v_lower, 
                    side_l.v_upper, 
                    side_l.v_lower, 
                    self.h, 
                    t_guess
                ))
            i += 1
        return thresholds
    
    def compute_thresholds_vectorized(self, valuations, side_k, side_l, separating_range_k, separating_range_l, N_grid=20, coeffs=None):
        """
        Compute thresholds for a vector of valuations, using vectorized root-finding
        in the separating range, and scalar logic for the rest.
    
        :param valuations: Vector of valuations for side_k.
        :return: Vector of computed thresholds.
        """
        thresholds = np.full_like(valuations, side_l.v_upper)  # Initialize all to upper bound initially
    
        # Determine indices where valuations fall within the separating range
        separating_indices = (valuations >= min(separating_range_k)) & (valuations <= max(separating_range_k))
        valuations_in_range = valuations[separating_indices]

        if len(valuations_in_range) == 0:
            # No valuations in the separating range
            return thresholds
        # Check if the separating range has collapsed
        if np.abs(max(separating_range_k)-min(separating_range_k))<1e-8:
            # Separating range is a single point
            # All thresholds within the separating range are constant
            thresholds[separating_indices] = side_l.v_upper  # or appropriate constant value
            return thresholds
    
        # Create grid points within the separating range
        if N_grid != None:
            v_k_grid = np.linspace(min(separating_range_k), max(separating_range_k), N_grid)
        else:
            v_k_grid = valuations_in_range
            
        if coeffs is not None:
            if isinstance(coeffs, CubicSpline):
                t_guesses = coeffs(v_k_grid)
            elif isinstance(coeffs, np.ndarray):
                t_guesses = np.polyval(coeffs, v_k_grid)
            else:
                raise ValueError("Unrecognized coefficient type. Must be CubicSpline or array.")
        else:
            # Use midpoint of side_l's valuation range as initial guess
            t_guesses = np.full_like(v_k_grid, (side_l.v_lower + side_l.v_upper) / 2)

    
        # Compute thresholds using the vectorized method for those within the separating range
        thresholds_grid = compute_thresholds_vector_min(
            v_k_grid,
            side_k.F,
            side_k.f,
            side_l.F,
            side_l.f,
            side_k.v_upper,
            side_k.v_lower,
            separating_range_l[1],
            separating_range_l[0],
            self.h,
            t_guesses
        )
    
        # Create an interpolation function
        interpolation_func = CubicSpline(v_k_grid, thresholds_grid)
    
        # Interpolate thresholds for valuations within the separating range
        thresholds[separating_indices] = interpolation_func(valuations_in_range)
    
        # For valuations above or below the separating range, thresholds are just the bounds
        thresholds[valuations < min(separating_range_k)] = side_l.v_upper
        thresholds[valuations > max(separating_range_k)] = side_l.v_lower
    
        return thresholds

    def compute_thresholds(self, N_grid = 25,  buyers_threshold_spline=None, sellers_threshold_spline=None):       
        
        # Start time measurement
        start_time = time.time()
        
        if buyers_threshold_spline is not None and sellers_threshold_spline is not None:
            # Use the provided splines for threshold computation
            self.buyers_thresholds = self.compute_thresholds_vectorized(
                self.buyers.get_valuations(), 
                self.buyers, 
                self.sellers, 
                self.buyers_separating_range,
                self.sellers_separating_range,
                N_grid=N_grid,
                coeffs=buyers_threshold_spline
            )
            
            self.sellers_thresholds = self.compute_thresholds_vectorized(
                self.sellers.get_valuations(), 
                self.sellers, 
                self.buyers, 
                self.sellers_separating_range, 
                self.buyers_separating_range,
                N_grid=N_grid,
                coeffs=sellers_threshold_spline
            )
        else:
            
            if (self.buyers_separating_range[1] == self.buyers_separating_range[0]) and self.exclusion_buyers:
                # Create an array of thresholds for buyers, filled with the constant value
             
                self.buyers_thresholds = np.full(len(self.buyers.valuations), self.sellers_separating_range[1])
            else:
                buyers_coeffs = compute_cubic_spline_guess(
                    self.buyers_separating_range, 
                    self.sellers_separating_range, 
                    self.buyers.F, 
                    self.buyers.f, 
                    self.sellers.F, 
                    self.sellers.f, 
                    self.buyers.v_upper, 
                    self.buyers.v_lower, 
                    self.sellers.v_upper, 
                    self.sellers.v_lower, 
                    self.h
                    )
                # Pass the coefficients to the threshold computation
                self.buyers_thresholds = self.compute_thresholds_vectorized(
                    self.buyers.get_valuations(), 
                    self.buyers, 
                    self.sellers, 
                    self.buyers_separating_range,
                    self.sellers_separating_range,
                    N_grid = N_grid,
                    coeffs=buyers_coeffs
                )
            if (self.sellers_separating_range[1] == self.sellers_separating_range[0]) and self.exclusion_sellers:
                # Create an array of thresholds for sellers, filled with the constant value
              
                self.sellers_thresholds = np.full(len(self.sellers.valuations), self.buyers_separating_range[1])
            else:
                sellers_coeffs = compute_cubic_spline_guess(
                    self.sellers_separating_range, 
                    self.buyers_separating_range, 
                    self.sellers.F, 
                    self.sellers.f, 
                    self.buyers.F, 
                    self.buyers.f, 
                    self.sellers.v_upper, 
                    self.sellers.v_lower, 
                    self.buyers.v_upper, 
                    self.buyers.v_lower, 
                    self.h
                )
                
                self.sellers_thresholds = self.compute_thresholds_vectorized(
                    self.sellers.get_valuations(), 
                    self.sellers, 
                    self.buyers, 
                    self.sellers_separating_range, 
                    self.buyers_separating_range,
                    N_grid = N_grid,
                    coeffs=sellers_coeffs
                )
        
        end_time = time.time()
        print(f"Threshold computation runtime: {end_time - start_time:.4f} seconds", flush=True)
    
    def generate_threshold_splines(self):
            """
            Generate cubic spline approximations for buyers' and sellers' thresholds based on the current simulation.
            
            Returns:
            - tuple:
                - buyers_threshold_spline (CubicSpline): Spline approximation for buyers' thresholds.
                - sellers_threshold_spline (CubicSpline): Spline approximation for sellers' thresholds.
            """
            # Extract valuations and thresholds for buyers and sellers
            (buyers_valuations, buyers_thresholds), (sellers_valuations, sellers_thresholds) = self.get_valuations_and_thresholds()
            
            # Ensure that there are enough points to fit a spline
            if len(buyers_valuations) < 4 or len(sellers_valuations) < 4:
                raise ValueError("Not enough points to fit a cubic spline. At least 4 points are required.", flush=True)
            
            # Fit cubic splines
            buyers_threshold_spline = CubicSpline(buyers_valuations, buyers_thresholds, bc_type='clamped')
            sellers_threshold_spline = CubicSpline(sellers_valuations, sellers_thresholds, bc_type='clamped')
            
            return buyers_threshold_spline, sellers_threshold_spline
    
    def calculate_matches(self, side_k, side_l, thresholds_k):
        matches = {}
        data_k = side_k.data
        data_l = side_l.data
        for i, v_k in enumerate(data_k['Valuation']):
            t_k = thresholds_k[i]
            agent_id_k = data_k['ID'][i]
            # Find indices of side_l where valuations > t_k
            matched_indices = np.where(data_l['Valuation'] > t_k)[0]
            # Get the IDs of the matched agents from side_l
            matched_IDs = data_l['ID'][matched_indices]
            matches[agent_id_k] = matched_IDs.tolist()
        return matches

    def compute_matches(self):
        self.buyers_matches = self.calculate_matches(self.buyers, self.sellers, self.buyers_thresholds)
        self.sellers_matches = self.calculate_matches(self.sellers, self.buyers, self.sellers_thresholds)

    
    def calculate_payments(self, v_k, t_k, side_k, side_l, p_lower_l):
        return compute_payments_from_envelope(
            v_k, t_k, side_k.F, side_k.f, side_l.F, side_l.f,
            side_k.v_upper, side_k.v_lower, side_l.v_upper, side_l.v_lower, p_lower_l
        )
    
    def compute_payments(self):
        v_values_buyers = self.buyers.get_valuations()
        v_values_sellers = self.sellers.get_valuations()
    
        # Compute the first payment element for buyers and sellers using payment_lower function
        first_payment_buyer = compute_payment_lower(
            self.buyers.v_lower,
            self.sellers.v_upper,
            self.sellers.F,
            self.sellers_thresholds[0]
        )
        first_payment_seller = compute_payment_lower(
            self.sellers.v_lower,
            self.buyers.v_upper,
            self.buyers.F,
            self.buyers_thresholds[0]
        )
    
        # Compute the payments using the optimized function
        self.buyers_payments = compute_payments_from_envelope(
            v_values_buyers,
            self.buyers_thresholds,
            self.buyers.F,
            self.buyers.f,
            self.sellers.F,
            self.sellers.f,
            self.buyers.v_upper,
            self.buyers.v_lower,
            self.sellers.v_upper,
            self.sellers.v_lower,
            first_payment_buyer
        )
    
        self.sellers_payments = compute_payments_from_envelope(
            v_values_sellers,
            self.sellers_thresholds,
            self.sellers.F,
            self.sellers.f,
            self.buyers.F,
            self.buyers.f,
            self.sellers.v_upper,
            self.sellers.v_lower,
            self.buyers.v_upper,
            self.buyers.v_lower,
            first_payment_seller
        )

    def compute_theoretical_values(self):
        min_valuation_buyers = self.buyers.v_lower
        max_valuation_buyers = self.buyers.v_upper
        min_valuation_sellers = self.sellers.v_lower
        max_valuation_sellers = self.sellers.v_upper

        
        self.theoretical_valuations_buyers = np.linspace(min_valuation_buyers, max_valuation_buyers,200)
        self.theoretical_valuations_sellers = np.linspace(min_valuation_sellers, max_valuation_sellers, 200)

       # Compute theoretical thresholds using the shared method
       
       
        buyers_coeffs = compute_cubic_spline_guess(
        self.buyers_separating_range, 
        self.sellers_separating_range, 
        self.buyers.F, 
        self.buyers.f, 
        self.sellers.F, 
        self.sellers.f, 
        self.buyers.v_upper, 
        self.buyers.v_lower, 
        self.sellers.v_upper, 
        self.sellers.v_lower, 
        self.h
        )
    
        sellers_coeffs = compute_cubic_spline_guess(
            self.sellers_separating_range, 
            self.buyers_separating_range, 
            self.sellers.F, 
            self.sellers.f, 
            self.buyers.F, 
            self.buyers.f, 
            self.sellers.v_upper, 
            self.sellers.v_lower, 
            self.buyers.v_upper, 
            self.buyers.v_lower, 
            self.h
        )
        
            
        self.theoretical_thresholds_buyers = self.compute_thresholds_vectorized(self.theoretical_valuations_buyers, self.buyers, self.sellers, self.buyers_separating_range, self.sellers_separating_range, buyers_coeffs)
        self.theoretical_thresholds_sellers = self.compute_thresholds_vectorized(self.theoretical_valuations_sellers, self.sellers, self.buyers, self.sellers_separating_range, self.buyers_separating_range, sellers_coeffs)
        
        # Calculate lower payments for buyers and sellers using payment_lower function
        lower_payment_buyer = compute_payment_lower(
            self.buyers.v_lower,
            self.sellers.v_upper,
            self.sellers.F,
            self.theoretical_thresholds_sellers[0]
        )
        lower_payment_seller = compute_payment_lower(
            self.sellers.v_lower,
            self.buyers.v_upper,
            self.buyers.F,
            self.theoretical_thresholds_buyers[0]
        )
     
        # Vectorized computation of theoretical payments for buyers
        # self.theoretical_payments_buyers = compute_payments_from_envelope(
        #     self.theoretical_valuations_buyers,
        #     np.array(self.theoretical_thresholds_buyers),
        #     self.buyers.F,
        #     self.buyers.f,
        #     self.sellers.F,
        #     self.sellers.f,
        #     self.buyers.v_upper,
        #     self.buyers.v_lower,
        #     self.sellers.v_upper,
        #     self.sellers.v_lower,
        #     lower_payment_buyer
        # )
    
        # # Vectorized computation of theoretical payments for sellers
        # self.theoretical_payments_sellers = compute_payments_from_envelope(
        #     self.theoretical_valuations_sellers,
        #     np.array(self.theoretical_thresholds_sellers),
        #     self.sellers.F,
        #     self.sellers.f,
        #     self.buyers.F,
        #     self.buyers.f,
        #     self.sellers.v_upper,
        #     self.sellers.v_lower,
        #     self.buyers.v_upper,
        #     self.buyers.v_lower,
        #     lower_payment_seller
        # )
            
    def plot_buyers_payments(self):
        v_values_buyers = self.buyers.get_valuations()

        plt.figure(figsize=(10, 6))
        #plt.plot(self.theoretical_valuations_buyers, self.theoretical_payments_buyers, label='Theoretical Payments Buyers')
        plt.plot(v_values_buyers, self.buyers_payments, label='Empirical Payments Buyers', alpha=0.5)
        plt.xlabel('Valuation (v)')
        plt.ylabel('Payments')
        plt.title('Payments vs. Valuation for Buyers')
        plt.legend()
        plt.show()

    def plot_sellers_payments(self):
        v_values_sellers = self.sellers.get_valuations()

        plt.figure(figsize=(10, 6))
        #plt.plot(self.theoretical_valuations_sellers, self.theoretical_payments_sellers, label='Theoretical Payments Sellers')
        plt.plot(v_values_sellers, self.sellers_payments, label='Empirical Payments Sellers', alpha=0.5)
        plt.xlabel('Valuation (v)')
        plt.ylabel('Payments')
        plt.title('Payments vs. Valuation for Sellers')
        plt.legend()
        plt.show()

    def plot_buyers_threshold_function(self):
        v_values_buyers = self.buyers.get_valuations()

        plt.plot(self.theoretical_valuations_buyers, self.theoretical_thresholds_buyers, label='Theoretical Thresholds Buyers')
        plt.scatter(v_values_buyers, self.buyers_thresholds, label='Empirical Thresholds Buyers', alpha=0.5)
        plt.xlabel('Valuation (v)')
        plt.ylabel('Threshold (t)')
        plt.title('Threshold Function vs. Valuation for Buyers')
        plt.legend()
        plt.show()

    def plot_sellers_threshold_function(self):
        v_values_sellers = self.sellers.get_valuations()

        plt.figure(figsize=(10, 6))
        plt.plot(self.theoretical_valuations_sellers, self.theoretical_thresholds_sellers, label='Theoretical Thresholds Sellers')
        plt.scatter(v_values_sellers, self.sellers_thresholds, label='Empirical Thresholds Sellers', alpha=0.5)
        plt.xlabel('Valuation (v)')
        plt.ylabel('Threshold (t)')
        plt.title('Threshold Function vs. Valuation for Sellers')
        plt.legend()

    def plot_matches(self, colors=('blue', 'orange'), thresholds="empirical", save_path=None):
       fig, axs = plt.subplots(1, 2, figsize=(20, 6))
       
       # Slightly darker shades for lines corresponding to the scatter plot colors
       line_colors = [mcolors.to_rgba(colors[0], alpha=0.8), mcolors.to_rgba(colors[1], alpha=0.8)]
       
       # Create ID-to-Valuation mappings
       buyer_id_to_valuation = {buyer_id: v for buyer_id, v in zip(self.buyers.data['ID'], self.buyers.get_valuations())}
       seller_id_to_valuation = {seller_id: v for seller_id, v in zip(self.sellers.data['ID'], self.sellers.get_valuations())}
       
       # --- Plot for Buyers ---
       buyers_v_k = []
       buyers_v_l = []
       
       for buyer in self.buyers.data:
           buyer_id = buyer['ID']
           v_k = buyer_id_to_valuation[buyer_id]
           matched_sellers = self.buyers_matches.get(buyer_id, [])
           for seller_id in matched_sellers:
               v_l = seller_id_to_valuation.get(seller_id, None)
               if v_l is not None:
                   buyers_v_k.append(v_k)
                   buyers_v_l.append(v_l)
       
       # Scatter all buyers' matches at once
       axs[0].scatter(buyers_v_k, buyers_v_l, color=colors[0], alpha=0.5, s=10)
       axs[0].set_ylim(self.sellers.v_lower * 0.9, self.sellers.v_upper * 1.1)
       
       # Plot Thresholds
       if thresholds == "empirical":
           sorted_indices_buyers = np.argsort(self.buyers.get_valuations())
           sorted_valuations_buyers = np.array(self.buyers.get_valuations())[sorted_indices_buyers]
           sorted_thresholds_buyers = self.buyers_thresholds[sorted_indices_buyers]
           axs[0].plot(sorted_valuations_buyers, sorted_thresholds_buyers, label=r'$t_A(v_A)$', color=line_colors[0])
       else:
           axs[0].plot(self.theoretical_valuations_buyers, self.theoretical_thresholds_buyers, label=r'$t_A(v_A)$', color=line_colors[0])
       
       # Vertical and Horizontal Separation Lines
       axs[0].axvline(x=self.buyers_omega, color='black', linestyle='--', alpha=0.5)
       axs[0].axhline(y=self.sellers_omega, color='black', linestyle='--', alpha=0.5)
       
       # Add Text Labels for Separation Points
       axs[0].text(self.buyers_separating_range[0], axs[0].get_ylim()[0], r'$\omega_A$', color='black', verticalalignment='bottom', horizontalalignment='right')
       if self.bunching_buyers:
           axs[0].axvline(x=self.buyers_separating_range[1], color='black', linestyle='--', alpha=0.5)
           axs[0].text(self.buyers_separating_range[1], axs[0].get_ylim()[0], r'$t_B(\omega_B)$', color='black', verticalalignment='bottom', horizontalalignment='right')
       axs[0].text(axs[0].get_xlim()[0], self.sellers_omega, r'$\omega_B$', color='black', verticalalignment='top', horizontalalignment='left')
       
       axs[0].set_title("Side A's Matching Rule $\mathbf{s_A}$")
       axs[0].set_xlabel(r"$v_A$")
       axs[0].set_ylabel(r"$v_B$")
       axs[0].legend(loc='upper right')
       
       # --- Plot for Sellers ---
       sellers_v_l = []
       sellers_v_k = []
       
       for seller in self.sellers.data:
           seller_id = seller['ID']
           v_l = seller_id_to_valuation[seller_id]
           matched_buyers = self.sellers_matches.get(seller_id, [])
           for buyer_id in matched_buyers:
               v_k = buyer_id_to_valuation.get(buyer_id, None)
               if v_k is not None:
                   sellers_v_l.append(v_l)
                   sellers_v_k.append(v_k)
       
       # Scatter all sellers' matches at once
       axs[1].scatter(sellers_v_l, sellers_v_k, color=colors[1], alpha=0.5, s=10)
       axs[1].set_ylim(self.buyers.v_lower * 0.9, self.buyers.v_upper * 1.1)
       
       
       # Plot Thresholds
       if thresholds == "empirical":
           sorted_indices_sellers = np.argsort(self.sellers.get_valuations())
           sorted_valuations_sellers = np.array(self.sellers.get_valuations())[sorted_indices_sellers]
           sorted_thresholds_sellers = self.sellers_thresholds[sorted_indices_sellers]
           axs[1].plot(sorted_valuations_sellers, sorted_thresholds_sellers, label=r'$t_B(v_B)$', color=line_colors[1])
       else:
           axs[1].plot(self.theoretical_valuations_sellers, self.theoretical_thresholds_sellers, label=r'$t_B(v_B)$', color=line_colors[1])
       
       # Vertical and Horizontal Separation Lines
       axs[1].axvline(x=self.sellers_omega, color='black', linestyle='--', alpha=0.5)
       axs[1].axhline(y=self.buyers_omega, color='black', linestyle='--', alpha=0.5)
       
       # Add Text Labels for Separation Points
       axs[1].text(self.sellers_separating_range[0], axs[1].get_ylim()[0], r'$\omega_B$', color='black', verticalalignment='bottom', horizontalalignment='right')
       if self.bunching_sellers:
           axs[1].axvline(x=self.sellers_separating_range[1], color='black', linestyle='--', alpha=0.5)
           axs[1].text(self.sellers_separating_range[1], axs[1].get_ylim()[0], r'$t_A(\omega_A)$', color='black', verticalalignment='bottom', horizontalalignment='right')
       axs[1].text(axs[1].get_xlim()[0], self.buyers_omega, r'$\omega_A$', color='black', verticalalignment='top', horizontalalignment='left')
       
       axs[1].set_title("Side B's Matching Rule $\mathbf{s_B}$")
       axs[1].set_xlabel(r"$v_B$")
       axs[1].set_ylabel(r"$v_A$")
       axs[1].legend(loc='upper right')
       
       # Save or display the plot
       if save_path:
           plt.savefig(save_path, format='pdf', bbox_inches='tight')
           print(f"Plot saved to {save_path}")
       else:
           plt.show()

    def plot_payments(self, colors=('blue', 'orange'), save_path=None):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
        # Slightly darker shades for lines corresponding to the scatter plot colors
        line_colors = [mcolors.to_rgba(colors[0], alpha=0.8), mcolors.to_rgba(colors[1], alpha=0.8)]
    
        # Set x axis limits with 10% padding
        axs[0].set_xlim(self.buyers.v_lower * 0.9, self.buyers.v_upper * 1.1)
        axs[1].set_xlim(self.sellers.v_lower * 0.9, self.sellers.v_upper * 1.1)
        
        
    
        # Plot for buyers' payments
        v_values_buyers = self.buyers.get_valuations()
        axs[0].scatter(v_values_buyers, self.buyers_payments, alpha=0.5, color=colors[0])
        axs[0].plot(v_values_buyers, self.buyers_payments, color=line_colors[0])
    
        if self.bunching_sellers:
            axs[0].axvline(x=self.buyers_threshold_omega, color='black', linestyle='--', alpha=0.5)
            axs[0].text(self.buyers_threshold_omega, axs[0].get_ylim()[0], r'$t_A(\omega_A)$', color='black', verticalalignment='bottom', horizontalalignment='right')
        else:
            axs[0].axvline(x=self.buyers_omega, color='black', linestyle='--', alpha=0.5)
            axs[0].text(self.buyers_omega, axs[0].get_ylim()[0], r'$\omega_A$', color='black', verticalalignment='bottom', horizontalalignment='right')
    
        axs[0].set_xlabel(r"$v_A$")
        axs[0].set_ylabel(r"$p_A(v_A)$")
        axs[0].set_title("Side A's payment rule $p_A$")
    
        # Plot for sellers' payments
        v_values_sellers = self.sellers.get_valuations()
        axs[1].scatter(v_values_sellers, self.sellers_payments, alpha=0.5, color=colors[1])
        axs[1].plot(v_values_sellers, self.sellers_payments, color=line_colors[1])
    
        if self.bunching_buyers:
            axs[1].axvline(x=self.sellers_threshold_omega, color='black', linestyle='--', alpha=0.5)
            axs[1].text(self.sellers_threshold_omega, axs[1].get_ylim()[0], r'$t_B(\omega_B)$', color='black', verticalalignment='bottom', horizontalalignment='right')
        else:
            axs[1].axvline(x=self.sellers_omega, color='black', linestyle='--', alpha=0.5)
            axs[1].text(self.sellers_omega, axs[1].get_ylim()[0], r'$\omega_B$', color='black', verticalalignment='bottom', horizontalalignment='right')
    
        axs[1].set_xlabel(r"$v_B$")
        axs[1].set_ylabel(r"$p_B(v_B)$")
        axs[1].set_title("Side B's payment rule $p_B$")
    
        # Save or display the plot
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()