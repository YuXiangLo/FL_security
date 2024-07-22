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
from functools import reduce
import numpy as np
import flwr as fl

def krum_aggregate(results: List[Tuple[NDArrays, int]], num_selected: int = None) -> NDArrays:
    """Aggregate using the Krum method."""
    n = len(results)
    if num_selected is None:
        num_selected = n - 2  # Usually n - 2 to tolerate 1 Byzantine client

    # Extract weights
    client_updates = [weights for weights, _ in results]

    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = np.linalg.norm(
                np.concatenate([layer.flatten() for layer in client_updates[i]]) - 
                np.concatenate([layer.flatten() for layer in client_updates[j]])
            )

    scores = np.zeros(n)
    # Compute scores for each client update
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        scores[i] = np.sum(sorted_distances[:num_selected])

    # Select the update with the lowest score
    selected_index = np.argmin(scores)
    return client_updates[selected_index]

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
        aggregated_ndarrays = krum_aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated