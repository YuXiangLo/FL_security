# no defense: 7 epoch --> test acc = 0.4928
from typing import Dict, List, Optional, Tuple, Union
from logging import WARNING
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
import numpy as np
import flwr as fl
from functools import reduce

# import cvxpy as cp
from scipy.optimize import minimize

# Parameters
NUM_SAMPLED_DIMENSION = 6000
LAMBDA = 0.1
MINVAR_EXPONENT = 5

def weighted_MinVar_aggregate(results: List[Tuple[NDArrays, int]], num_selected: int = None) -> NDArrays:
    """Aggregate using the MinVar method."""

    # Extract weights

    # List of NDArrays
    client_updates = [weights for weights, _ in results]
    num_clients = len(client_updates)

    # https://chatgpt.com/share/be761e37-7268-4322-a8bc-1e00208cee31

    # aggregation weight for each client
    print("Number of clients = " + str(num_clients))

    flattened_client_update_list = []
    for client_update in client_updates:
        flattened_client_update = []
        for layer_id, layer in enumerate(client_update):
            flattened_layer = layer.flatten()
            # norm = np.linalg.norm(flattened_layer, axis=0, keepdims=True)
            # flattened_layer = (flattened_layer / norm)
            flattened_client_update.extend(flattened_layer)
        flattened_client_update_list.append(np.array(flattened_client_update))

    sampled_dimension = np.random.choice(len(flattened_client_update_list[0]), NUM_SAMPLED_DIMENSION, replace=False)
    flattened_client_update_list = [flattened_client_update_list[i][sampled_dimension] for i in range(num_clients)]

    # Objective function
    def objective(h, updates):
        variances = []
        update_size = len(updates[0])
        for i in range(update_size):
            updates_i = np.array([updates[j][i] for j in range(num_clients)])
            original_mean = np.mean(updates_i)
            original_std = np.std(updates_i)
            if original_std == 0:
                original_std = 1
            updates_i = (updates_i - original_mean) / original_std
            weighted_updates = updates_i * h
            weighted_mean = np.sum(weighted_updates) / num_clients
            weighted_minus = h * (weighted_updates - weighted_mean)
            weighted_variance = np.sum(weighted_minus**2)
            variances.append(weighted_variance)

        variance_term = np.sum(variances)
        l2_term = np.linalg.norm(h, 2) * LAMBDA
        return variance_term + l2_term

    # Constraint: weights sum to 1
    def constraint_sum_to_one(h):
        return np.sum(h) - 1

    # Constraint: weights are non-negative
    def constraint_non_negative(h):
        return h

    # Initial guess for the weights
    initial_guess = np.ones(num_clients) / num_clients

    # Constraints for scipy.optimize
    constraints = [
        {'type': 'eq', 'fun': constraint_sum_to_one},
        {'type': 'ineq', 'fun': constraint_non_negative}
    ]

    # Bounds for the variables (weights between 0 and 1)
    bounds = [(0, 1) for _ in range(num_clients)]

    # Solve the optimization problem
    result = minimize(objective, initial_guess, args=(flattened_client_update_list,), method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimal weights
    optimal_weights = np.array(result.x)
    optimal_weights = np.power(optimal_weights, MINVAR_EXPONENT)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    print("Optimal Weights:", optimal_weights)
    weighted_updates = [
        [layer * optimal_weights[i] for layer in client_updates[i]] for i in range(num_clients)
    ]

    aggregated_updates: NDArrays = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_updates)
    ]

    return aggregated_updates


class Robust_Server(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Krum aggregation."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # aggregated_ndarrays = minvar_aggregate(weights_results)
        aggregated_ndarrays = weighted_MinVar_aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
