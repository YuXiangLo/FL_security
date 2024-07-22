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

import cvxpy as cp

NUM_SAMPLED_DIMENSION = 5000
LAMBDA = 0.01
MINVAR_EXPONENT = 5

def minvar_aggregate(results: List[Tuple[NDArrays, int]], num_selected: int = None) -> NDArrays:
    """Aggregate using the MinVar method."""

    # Extract weights

    # List of NDArrays
    client_updates = [weights for weights, _ in results]
    num_clients = len(client_updates)

    # https://chatgpt.com/share/be761e37-7268-4322-a8bc-1e00208cee31

    # aggregation weight for each client
    print("Number of clients = " + str(num_clients))

    # sampled_layer_num = np.random.choice(len(client_updates[0]), len(client_updates[0]) // 2, replace=False)
    flattened_client_update_list = []
    for client_update in client_updates:
        flattened_client_update = []
        for layer_id, layer in enumerate(client_update):
            # if layer_id not in sampled_layer_num:
            #     continue
            flattened_client_update.extend(layer.flatten())
        flattened_client_update_list.append(np.array(flattened_client_update))

    sampled_dimension = np.random.choice(len(flattened_client_update_list[0]), NUM_SAMPLED_DIMENSION, replace=False)
    flattened_client_update_list = [flattened_client_update_list[i][sampled_dimension] for i in range(num_clients)]

    h = cp.Variable(num_clients)
    variances = []
    # i is the index of dimension
    for i in range(len(flattened_client_update_list[0])):
        updates_i = [flattened_client_update_list[j][i] for j in range(num_clients)]
        if(i == 0): print("Updates Format: ", updates_i)
        weighted_updates = cp.hstack([updates_i[j] * h[j] for j in range(num_clients)])
        weighted_mean = cp.sum(weighted_updates) / num_clients
        weighted_variance = cp.sum_squares(weighted_updates - weighted_mean)
        variances.append(weighted_variance)
    variance_term = cp.sum(variances)

    l2_term = cp.norm(h, 2) * LAMBDA

    # Objective function
    objective = cp.Minimize(variance_term + l2_term)

    # Constraints
    constraints = [cp.sum(h) == 1, h >= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    print("Solving QP ...")
    prob.solve()
   

    # Increasing the affect of MinVar
    h_value = h.value
    h_value = np.power(h_value, MINVAR_EXPONENT)
    # Normalize the transformed weights to sum to 1
    h_value = h_value / np.sum(h_value)

    print("Optimal Weights: ", h_value)
    weighted_updates = [
        [layer * h_value[i] for layer in client_updates[i]] for i in range(num_clients)
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
        aggregated_ndarrays = minvar_aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
