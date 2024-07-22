import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Parameters, NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.server import FitRes
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def multi_krum_aggregate(results: List[Tuple[NDArrays, int]], num_malicious: int, krum_k: int) -> NDArrays:
    n = len(results)
    if n <= num_malicious:
        raise ValueError("Number of clients must be greater than the number of malicious clients")

    # Extract weights
    weights = [res[0] for res in results]

    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = np.linalg.norm(np.concatenate([layer.flatten() for layer in weights[i]]) - 
                                             np.concatenate([layer.flatten() for layer in weights[j]]))
            distances[j, i] = distances[i, j]

    # Compute scores
    scores = np.zeros(n)
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        scores[i] = np.sum(sorted_distances[:n - num_malicious - 2])

    # Select the top `krum_k` clients with the smallest scores
    krum_indices = np.argsort(scores)[:krum_k]
    selected_weights = [weights[i] for i in krum_indices]

    # Compute the average of the selected weights
    aggregated_weights = []
    for layer_index in range(len(selected_weights[0])):
        layer_weights = np.array([weights[layer_index] for weights in selected_weights])
        mean_layer_weights = np.mean(layer_weights, axis=0)
        aggregated_weights.append(mean_layer_weights)

    return aggregated_weights

class MultiKrumServer(FedAvg):
    def __init__(self, num_malicious: int, krum_k: int, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious
        self.krum_k = krum_k

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = multi_krum_aggregate(weights_results, self.num_malicious, self.krum_k)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log.warning("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
