import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Parameters, NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, NDArray
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.server import FitRes
import logging
from functools import reduce

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
from defense_method.aggregate import aggregate_trimmed_avg

class TrimServer(FedAvg):
    def __init__(self, trim_ratio: int = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio

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

        aggregated_ndarrays = aggregate_trimmed_avg(weights_results, self.trim_ratio)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log.warning("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
