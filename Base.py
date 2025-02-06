#!/usr/bin/env python3
# -*- coding: utf-8 -*-|
"""
Created on Sat Jul 27 16:59:17 2024

@author: michelev
"""

import numpy as np
import pandas as pd
import json
from scipy.optimize import fsolve, minimize, root
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.stats import truncnorm
from sklearn.preprocessing import MinMaxScaler

#%%

def normalize_columns(df, columns_to_normalize):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])
    normalized_df = pd.DataFrame(normalized_data, columns=[col + '_norm' for col in columns_to_normalize], index=df.index)
    return pd.concat([df, normalized_df], axis=1)

def save_optimize_result_to_json(result, filename, exclude_attrs=None):
    """
    Save OptimizeResult to a JSON file, excluding specified attributes.
    
    Parameters:
    - result (OptimizeResult): The optimization result object.
    - filename (str): The name of the JSON file to save.
    - exclude_attrs (list of str): List of attribute names to exclude from serialization.
    """
    serializable_dict = optimize_result_to_serializable_dict(result, exclude_attrs)
    
    try:
        with open(filename, 'w') as json_file:
            json.dump(serializable_dict, json_file, indent=4)
        print(f"Optimization result successfully saved to {filename}")
    except Exception as e:
        print(f"Failed to save optimization result to JSON: {e}")
        
def optimize_result_to_serializable_dict(result, exclude_attrs=None):
    """
    Convert OptimizeResult to a dictionary with serializable types, excluding specified attributes.
    
    Parameters:
    - result (OptimizeResult): The optimization result object.
    - exclude_attrs (list of str): List of attribute names to exclude from serialization.
    
    Returns:
    - dict: A dictionary representation of the OptimizeResult suitable for serialization.
    """
    if exclude_attrs is None:
        exclude_attrs = []
    
    serializable_dict = {}
    
    for key, value in result.items():
        if key in exclude_attrs:
            continue  # Skip excluded attributes
        
        if isinstance(value, np.ndarray):
            serializable_dict[key] = value.tolist()
        elif isinstance(value, (bool, int, float, str)) or value is None:
            serializable_dict[key] = value
        elif isinstance(value, (dict, list, tuple)):
            serializable_dict[key] = value  # Assuming nested structures are JSON-serializable
        else:
            # For complex objects like hess_inv, convert to string or handle appropriately
            serializable_dict[key] = str(value)
    
    return serializable_dict

def perturb_duplicates(valuations, epsilon=1e-8):
    """
    Add a small deterministic perturbation to duplicate valuations to ensure uniqueness.

    Parameters:
    - valuations: np.ndarray, array of valuations
    - epsilon: float, magnitude of the perturbation

    Returns:
    - valuations_unique: np.ndarray, array of valuations with duplicates perturbed
    """
    valuations = valuations.copy()  # Avoid modifying the original array

    # **Efficiently check for duplicates**
    _, counts = np.unique(valuations, return_counts=True)
    if np.all(counts == 1):
        # No duplicates found; return the original valuations immediately
        return valuations

    # **Proceed to perturb duplicates only if duplicates are found**

    # Sort the valuations and keep track of the original indices
    sorted_indices = np.argsort(valuations)
    sorted_valuations = valuations[sorted_indices]

    # Initialize an array to hold the perturbations
    perturbations = np.zeros_like(valuations)

    # Initialize variables to keep track of duplicates
    i = 0
    n = len(valuations)

    while i < n:
        # Identify the current valuation
        current_valuation = sorted_valuations[i]

        # Find all indices where the valuation is equal to current_valuation
        j = i + 1
        duplicate_count = 1  # Count of duplicates, including the current one
        while j < n and sorted_valuations[j] == current_valuation:
            duplicate_count += 1
            j += 1

        # Apply deterministic perturbations to duplicates
        if duplicate_count > 1:
            # For duplicates, generate perturbations: epsilon*0, epsilon*1, epsilon*2, ...
            for k in range(duplicate_count):
                idx = i + k
                perturbations[sorted_indices[idx]] = epsilon * k  # First occurrence gets 0

        # Move to the next unique valuation
        i = j

    # Apply perturbations to valuations
    valuations += perturbations

    return valuations
    

def get_truncnorm_params(mean, left, right, scale):
    a = (left - mean) / scale
    b = (right - mean) / scale
    return a, b, mean, scale

def get_truncnorm_upper_bound(truncnorm_distribution):
    """
    Returns the largest value in the support of the truncated normal distribution.
    
    :param truncnorm_distribution: A scipy.stats.truncnorm distribution object.
    :return: The upper bound of the support.
    """
    # Extract the arguments from the distribution
    a, b, loc, scale = truncnorm_distribution.args
    return b * scale + loc

def get_truncnorm_lower_bound(truncnorm_distribution):
    """
    Returns the smallest value in the support of the truncated normal distribution.
    
    :param truncnorm_distribution: A scipy.stats.truncnorm distribution object.
    :return: The lower bound of the support.
    """
    # Extract the arguments from the distribution
    a, b, loc, scale = truncnorm_distribution.args
    return a * scale + loc

def get_truncnorm_range(truncnorm_distribution):
    
    value_range = np.array([get_truncnorm_lower_bound(truncnorm_distribution), get_truncnorm_upper_bound(truncnorm_distribution)])
    
    return value_range

def generate_predictors(N, K, means, lower_bounds, upper_bounds, scales):
    """
    Generate uncorrelated predictors with specified means using truncated normal distributions.

    :param N: Number of observations.
    :param K: Number of predictors.
    :param means: List or array of means for each predictor (length K).
    :param lower_bounds: List or array of lower bounds for each predictor (length K).
    :param upper_bounds: List or array of upper bounds for each predictor (length K).
    :param scales: List or array of scales (standard deviations) for each predictor (length K).
    :return: A NumPy array of shape (N, K) containing the generated predictors.
    """
    X = np.zeros((N, K))
    for j in range(K):
        a, b = (lower_bounds[j] - means[j]) / scales[j], (upper_bounds[j] - means[j]) / scales[j]
        X[:, j] = truncnorm(a, b, loc=means[j], scale=scales[j]).rvs(N)
    return X

def f_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def virtual_valuation(v, F, f, h, epsilon=1e-8):
    if h == "P":
        f_value = f(v)
        return v - (1 - F(v)) / np.maximum(f_value,epsilon)  # Adjust as needed for your case
    elif h == "W":
        return v
    
def virtual_valuation_derivative(v, F, f, h):
    if h == "P":
        f_prime = lambda x: f_derivative(f,x)
        return 2 + (1 - F(v)) * f_prime(v) / (f(v)**2)
    elif h == "W":
        return 1
    
def empirical_pdf(v_values, F_values, method = "central", epsilon=1e-10):
    """
    Compute the numerical derivative of the empirical CDF to get the PDF.
    
    :param v_values: Array of valuation points.
    :param F_values: Array of CDF values corresponding to v_values.
    :param epsilon: A small value added to prevent zero PDF values.
    :return: Array of PDF values corresponding to v_values.
    """
    sorted_indices = np.argsort(v_values)
    v_values = v_values[sorted_indices]
    F_values = F_values[sorted_indices]
    # Use central differences to approximate the derivative (PDF)
    f_values = np.zeros_like(F_values)
    
    # Choose method for interior points
    if method == 'central':
        # Central difference for interior points
        f_values[1:-1] = (F_values[2:] - F_values[:-2]) / (v_values[2:] - v_values[:-2])
    elif method == 'forward':
        # Forward difference for interior points
        f_values[1:-1] = (F_values[2:] - F_values[1:-1]) / (v_values[2:] - v_values[1:-1])
    elif method == 'backward':
        # Backward difference for interior points
        f_values[1:-1] = (F_values[1:-1] - F_values[:-2]) / (v_values[1:-1] - v_values[:-2])
    else:
        raise ValueError("Method must be 'central', 'forward', or 'backward'.")
    
    # Compute PDF values for boundaries
    f_values[0] = (F_values[1] - F_values[0]) / (v_values[1] - v_values[0])  # Forward difference at lower boundary
    f_values[-1] = (F_values[-1] - F_values[-2]) / (v_values[-1] - v_values[-2])  # Backward difference at upper boundary

    f_values = np.maximum(f_values, epsilon)
    # Normalize the PDF
    integral = np.trapz(f_values, v_values)
    f_values /= integral
    
    return f_values


def average_intensity_value(v, F, v_upper,epsabs=1.49e-8, epsrel=1.49e-8, limit=1000):
    lambda x: sigma_prime(x)*F(x)
    integral, error = quad(F, v, v_upper, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return sigma(v_upper) * F(v_upper) - sigma(v) * F(v) - integral + 1

def average_intensity_vectorized(v_vector, F, v_upper):
    """
    Optimized version of average_intensity using cumulative integration.
    
    :param v_vector: Vector of lower bounds for the integral.
    :param F: CDF function.
    :param v_upper: Upper bound of the integral.
    :return: Vector of average intensity values.
    """
    # Compute the cumulative integral from v_vector[0] to each point in v_vector
    # Note: cumtrapz integrates over the range [v[0], v[-1]]
    F_values = F(v_vector)
    sigma_prime_values = sigma_prime(v_vector)
    cumulative_integral = cumulative_trapezoid(sigma_prime_values * F_values, v_vector, initial=0)
    
    # Calculate average intensity using the cumulative integral
    # F(v_upper) * sigma(v_upper) is constant across all v_k
    average_intensity_values = sigma(v_upper) * F(v_upper) - sigma(v_vector) * F_values - cumulative_integral + 1
    
    return average_intensity_values
    
def average_intensity(v, F, v_upper, method="auto", epsabs=1.49e-2, epsrel=1.49e-4, limit=500):
    """
    Compute average intensity using either the scalar or vectorized approach.
    
    :param v: Scalar or vector of lower bounds for the integral.
    :param F: CDF function.
    :param v_upper: Upper bound of the integral.
    :param method: Method to use ("auto", "scalar", or "vectorized").
    :param epsabs: Absolute error tolerance for the integration.
    :param epsrel: Relative error tolerance for the integration.
    :param limit: Limit for the number of subdivisions in the integration.
    :return: Scalar or vector of average intensity values.
    """
    if method == "auto":
        if np.isscalar(v):
            return average_intensity_value(v, F, v_upper, epsabs=epsabs, epsrel=epsrel, limit=limit)
        else:
            return average_intensity_vectorized(v, F, v_upper)
    elif method == "scalar":
        if np.isscalar(v):
            return average_intensity_value(v, F, v_upper, epsabs=epsabs, epsrel=epsrel, limit=limit)
        else:
            raise ValueError("Input v is a vector, but 'scalar' method was specified.")
    elif method == "vectorized":
        if np.isscalar(v):
            raise ValueError("Input v is a scalar, but 'vectorized' method was specified.")
        else:
            return average_intensity_vectorized(v, F, v_upper)
    else:
        raise ValueError("Invalid method specified. Use 'auto', 'scalar', or 'vectorized'.")


def average_intensity_derivative(v, f):
    return -sigma(v)*f(v)

def euler_equation_long(v_k, v_l, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h):
    return f_k(v_k)*f_l(v_l)*(sigma(v_l)*virtual_valuation(v_k, F_k, f_k, h)/average_intensity(v_l, F_l, v_upper_l) + sigma(v_k)*virtual_valuation(v_l, F_l, f_l, h)/average_intensity(v_k, F_k, v_upper_k))

def euler_equation(v_k, v_l, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h):
    marginal_surplus_k = sigma(v_l)*virtual_valuation(v_k, F_k, f_k, h)*average_intensity(v_k, F_k, v_upper_k) 
    marginal_surplus_l = sigma(v_k)*virtual_valuation(v_l, F_l, f_l, h)*average_intensity(v_l, F_l, v_upper_l)
    return marginal_surplus_k + marginal_surplus_l

def penalized_euler_equation(v_k_vector, v_l, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, penalty=1e6):
    """
    Vectorized version of penalized Euler equation.

    :param v_k_vector: Vector of v_k values.
    :param v_l_vector: Vector of v_l values.
    :param F_k: CDF function for side k.
    :param f_k: PDF function for side k.
    :param F_l: CDF function for side l.
    :param f_l: PDF function for side l.
    :param v_upper_k: Upper bound for side k.
    :param v_lower_k: Lower bound for side k.
    :param v_upper_l: Upper bound for side l.
    :param v_lower_l: Lower bound for side l.
    :param h: Parameter for the Euler equation.
    :param penalty: Penalty value applied when v_l is out of bounds.
    :return: Vector of penalized Euler equation values.
    """
    # Check if v_l values are out of bounds and apply the penalty
    out_of_bounds = (v_l < v_lower_l) | (v_l > v_upper_l)
    
    # Calculate the Euler equation term where v_l is within bounds
    euler_term = euler_equation(v_k_vector, v_l, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)
    
    # Apply penalty to out-of-bounds elements
    euler_term[out_of_bounds] = penalty
    
    return euler_term

def euler_equation_derivative(v_k, v_l, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h):
    marginal_surplus_k = sigma(v_l)*(virtual_valuation_derivative(v_k, F_k, f_k, h)*average_intensity(v_k, F_k, v_upper_l)+virtual_valuation(v_k, F_k, f_k, h)*average_intensity_derivative(v_k,f_k))
    marginal_surplus_l = sigma_prime(v_k)*virtual_valuation(v_l, F_l, f_l, h)*average_intensity(v_l, F_l, v_upper_l)
    return marginal_surplus_k + marginal_surplus_l

def compute_linear_guess(separating_range_k, separating_range_l, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h):
    #midpoint_k = (separating_range_k[0] + separating_range_k[1]) / 2
    #midpoint_l = (separating_range_l[0] + separating_range_l[1]) / 2
    #t_midpoint = compute_threshold(midpoint_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, midpoint_l)
    
    # Fit a quadratic polynomial to these three points
    coeffs = np.polyfit([separating_range_k[0], separating_range_k[1]], [separating_range_l[1], separating_range_l[0]], 1)
    
    return coeffs

def compute_quadratic_guess(separating_range_k, separating_range_l, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h):
    midpoint_k = (separating_range_k[0] + separating_range_k[1]) / 2
    midpoint_l = (separating_range_l[0] + separating_range_l[1]) / 2
    t_midpoint = compute_threshold(midpoint_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, midpoint_l)
    
    # Fit a quadratic polynomial to these three points
    coeffs = np.polyfit([separating_range_k[0], midpoint_k, separating_range_k[1]], [separating_range_l[1], t_midpoint, separating_range_l[0]], 2)
    
    return coeffs

def compute_cubic_spline_guess(separating_range_k, separating_range_l, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h):
    # Calculate the original points in the separating range
    point1_k = separating_range_k[0]
    point2_k = separating_range_k[0] + 0.25 * (separating_range_k[1] - separating_range_k[0])
    point3_k = separating_range_k[0] + 0.5 * (separating_range_k[1] - separating_range_k[0])
    point4_k = separating_range_k[0] + 0.75 * (separating_range_k[1] - separating_range_k[0])
    point5_k = separating_range_k[1]
    
    point2_l = separating_range_l[1] - 0.25 * (separating_range_l[1] - separating_range_l[0])
    point3_l = separating_range_l[1] - 0.5 * (separating_range_l[1] - separating_range_l[0])
    point4_l = separating_range_l[1] - 0.75 * (separating_range_l[1] - separating_range_l[0])    
    
    # Calculate the additional points 5% closer to the bounds
    #point2_left_k = separating_range_k[0] + 0.05 * (separating_range_k[1] - separating_range_k[0])
    #point2_right_k = separating_range_k[1] - 0.05 * (separating_range_k[1] - separating_range_k[0])

    # Compute the thresholds at these points
    t1 = separating_range_l[1]
    t2 = compute_threshold(point2_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, point2_l)
    t3 = compute_threshold(point3_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, point3_l)
    t4 = compute_threshold(point4_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, point4_l)
    t5 = separating_range_l[0]
    
    #t2_left = compute_threshold(point2_left_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, point2_left_k)
    #t2_right = compute_threshold(point2_right_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, point2_right_k)

    # Fit a cubic spline to these points
    points_k = [point1_k, 
                #point2_left_k, 
                point2_k, point3_k, point4_k,
                #point2_right_k,
                point5_k]
    t_values = [t1,
                #t2_left,
                t2,
                t3,
                t4,
                #t2_right,
                t5]
    
    coeffs = CubicSpline(points_k, t_values)
    
    return coeffs

def compute_threshold(v_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, t_k_initial):
    # result = fsolve(
    #     lambda t_k: euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h),
    #     x0=t_k_initial,
    #     maxfev=500,
    #     #fprime=lambda t_k: euler_equation_derivative(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h),
    #     xtol=10e-10
    # )
    # return result[0]
    
    # def threshold_eq(t_k):
    #     return euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)

    solution = root(lambda t_k: penalized_euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h),
                    #fprime=lambda t_k: euler_equation_derivative(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h),
                    #col_deriv = True,
                    x0=t_k_initial, options = {'ftol': 10e-12}, method='lm')
    return solution.x[0]
    # if solution.success:
    #     return solution.x[0]
    # else:
    #     raise ValueError(f"Root finding failed: {solution.message}")
        
    
    # result = minimize(
    #     lambda t_k: euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)**2, 
    #     x0=t_k_initial,
    #     #jac=lambda t_k: 2 * euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h) *
    #     #                euler_equation_derivative(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h),
    #     method='L-BFGS-B',
    #     tol = 10e-8,
    #     bounds=[(v_lower_l, v_upper_l)]
    # )
    #return result.x[0]
    #return fsolve(lambda t_k: euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)**2, xtol=10e-6, x0=t_k_initial)[0]
    #]fprime = lambda t_k: euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)*euler_equation_derivative(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h), x0=t_k_initial)[0]
    #return fsolve(lambda t_k: euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h), fprime = lambda t_k: euler_equation_derivative(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h), x0=t_k_initial)[0]

def compute_thresholds_vector(v_k_vector, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, t_k_initial_vector):
    thresholds_eq = lambda t_k: penalized_euler_equation(v_k_vector, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h)
    threshold_prime_eq = lambda t_k: euler_equation_derivative(v_k_vector, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)
    solution = root(thresholds_eq, x0=t_k_initial_vector,
                    options = {'ftol': 1e-10, 'maxiter': 5000}, method='lm')
    
    #solution = root(thresholds_eq, x0=t_k_initial_vector,
    #                options = {'ftol': 10e-10}, method='broyden1')
    
    return solution.x
    
    #if solution.success:
    #    return solution.x
    #else:
    #    raise ValueError(f"Root finding failed: {solution.message}")
        
def compute_threshold_min(v_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, t_k_initial):
    """
    Compute thresholds for a vector of valuations using the minimize approach.
    
    :param v_k_vector: Vector of valuations for side_k.
    :param F_k: CDF function for side_k.
    :param f_k: PDF function for side_k.
    :param F_l: CDF function for side_l.
    :param f_l: PDF function for side_l.
    :param v_upper_k: Upper bound for side_k.
    :param v_lower_k: Lower bound for side_k.
    :param v_upper_l: Upper bound for side_l.
    :param v_lower_l: Lower bound for side_l.
    :param h: Discretization step.
    :param t_k_initial_vector: Initial guess for the thresholds.
    :return: Vector of computed thresholds.
    """
    
    def objective_function(t_k):
        # The objective function is the sum of squares of the penalized euler equations
        return euler_equation(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)**2

    # Minimize the objective function to find the thresholds
    solution = minimize(
        fun=objective_function,
        x0=t_k_initial,
        method='L-BFGS-B',  # Use a method suitable for bounded problems
        bounds=(v_lower_l, v_upper_l),  # Apply bounds to each element in the vector
        options = {'ftol': 1e-24}# Set a tolerance level
    )
    
    # Return the optimized threshold values
    return solution.x[0]

def compute_thresholds_vector_min(v_k_vector, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, h, t_k_initial_vector):
    """
    Compute thresholds for a vector of valuations using the minimize approach.
    
    :param v_k_vector: Vector of valuations for side_k.
    :param F_k: CDF function for side_k.
    :param f_k: PDF function for side_k.
    :param F_l: CDF function for side_l.
    :param f_l: PDF function for side_l.
    :param v_upper_k: Upper bound for side_k.
    :param v_lower_k: Lower bound for side_k.
    :param v_upper_l: Upper bound for side_l.
    :param v_lower_l: Lower bound for side_l.
    :param h: Discretization step.
    :param t_k_initial_vector: Initial guess for the thresholds.
    :return: Vector of computed thresholds.
    """
    
    def objective_function(t_k):
        # The objective function is the sum of squares of the penalized euler equations
        return np.sum(euler_equation(v_k_vector, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)**2)

    # Minimize the objective function to find the thresholds
    solution = minimize(
        fun=objective_function,
        x0=t_k_initial_vector,
        method='L-BFGS-B',  # Use a method suitable for bounded problems
        bounds=[(v_lower_l, v_upper_l)] * len(t_k_initial_vector),  # Apply bounds to each element in the vector
        options = {'ftol': 1e-24}# Set a tolerance level
    )
    
    # Return the optimized threshold values
    return solution.x

def compute_omega(F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, h):
    #omega_eq = lambda omega: (compute_threshold(omega, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h, v_upper_l/2) - v_upper_l)**2
    #omega_eq = lambda v_k: euler_equation(v_k, v_upper_l, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)
    #omega = fsolve(omega_eq, x0=v_upper_k/2, full_output=True)[0]
    #omega = fsolve(omega_eq, x0=v_lower_k, maxfev = 1000, xtol=10e-10)[0]
    
      
    def objective_function(t_k):
        # The objective function is the sum of squares of the penalized euler equations
        return euler_equation(t_k, v_upper_l, F_k, f_k, F_l, f_l, v_upper_k, v_upper_l, h)**2
    
    t_k_initial = (v_lower_k + v_upper_k)/2
    # Minimize the objective function to find the thresholds
    solution = minimize(
        fun=objective_function,
        x0=t_k_initial,
        method='L-BFGS-B',  # Use a method suitable for bounded problems
        bounds=[(v_lower_k, v_upper_k)],  # Apply bounds to each element in the vector
        options = {'ftol': 1e-24}# Set a tolerance level
    )
    
    # Return the optimized threshold values
    return solution.x[0]
    #solution = root(omega_eq,
    #                x0=(v_lower_k + v_upper_k)/2, options = {'ftol': 10e-10}, method='lm')
    #return solution.x[0]

    #omega = root(omega_eq, x0=v_upper_k, options = {'xtol': 10e-10, 'maxfev':500}, method='lm').x[0]
    #return omega

alpha = -0.5

def sigma(v):
    if alpha == 0:
        return v
    else:
        v = np.maximum(v,1e-10)
        return v**(1+alpha)
    
def sigma_prime(v):
    if alpha == 0:
        return 1
    else:
        v = np.maximum(v,1e-10)
        return (1+alpha)*v**alpha

def rho(x):
        return np.log(x)

def compute_payment_lower(v_lower_k, v_upper_l, F_l, t_l_lower):
        return v_lower_k*rho(average_intensity(t_l_lower, F_l, v_upper_l))
    
def compute_payments_from_envelope(v_k, t_k, F_k, f_k, F_l, f_l, v_upper_k, v_lower_k, v_upper_l, v_lower_l, p_lower_l):
    # Ensure v_lower_k is included in the vector of valuations
    if v_lower_k not in v_k:
        v_k_temp = np.insert(v_k, 0, v_lower_k)
        t_k_temp = np.insert(t_k, 0, v_upper_l)
    
    # Compute integral1 using the parallelized average_intensity function
    integral1 = v_k_temp * rho(average_intensity(t_k_temp, F_l, v_upper_l))
    
    # Compute rho(average_intensity) for the entire vector
    rho_values = rho(average_intensity(t_k_temp, F_l, v_upper_l))
    
    # Compute integral2 using cumulative trapezoidal integration from v_lower_k to each v_k
    integral2 = cumulative_trapezoid(rho_values, v_k_temp, initial=0)
    
    # Calculate payments
    payments = integral1[1:] - integral2[1:] - p_lower_l
    
    return payments

def chi(v, F, f, v_upper, h):
    
    if h == "P":
        chi = virtual_valuation(v, F, f, h)*average_intensity(v, F, v_upper)/sigma(v)
        #print(v-F(v))
    elif h == "W":
        chi = average_intensity(v, F, v_upper)
    
    return chi
            
def compute_and_check_increasing(v_values, chi_values):
    """
    Compute the numeric derivatives of chi with respect to v and check how many are positive.

    :param v_values: Array of v values.
    :param chi_values: Array of chi values corresponding to v_values.
    :return: Tuple containing the number of positive derivatives and the total number of derivatives.
    """
    derivatives = np.diff(chi_values) / np.diff(v_values)
    num_positive = np.sum(derivatives > 0)
    total = len(derivatives)
    return num_positive, total

def create_error_generator(distribution_name, params):
    """
    Reconstructs an error generator given the distribution name and parameters.

    :param distribution_name: Name of the distribution (e.g., 'truncnorm', 'norm').
    :param params: Dictionary of parameters for the distribution.
    :return: A scipy.stats distribution object.
    """
    from scipy.stats import truncnorm, norm  # Import necessary distributions

    if distribution_name == 'truncnorm':
        return truncnorm(*params)
    elif distribution_name == 'norm':
        return norm(*params)
    else:
        raise ValueError(f"Unsupported distribution: {distribution_name}")