from abc import ABC
from abc import abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import Union
from functools import cached_property

from bewer.metrics.dataset import Metric
from bewer.metrics.example import ExampleMetric

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example


class MetricCollection(object):
    """Collection of metrics for a dataset or an example.

    Attributes:
        src (Union[Dataset, Example]): The source object (dataset or example) to compute metrics for.
        metrics (list[Metric]): A list of valid Metric objects depending on the source object.
    """

    def __init__(self, _src: Union["Dataset", "Example"]):
        """Initialize the MetricCollection object.

        Args:
            src (Union[Dataset, Example]): The source object (dataset or example) to compute metrics for.
        """
        self._src = _src

    @cached_property
    def available_metrics(self) -> dict[str, type[Metric]]:
        """Get all available metrics (i.e., subclasses of Metric)."""
        # TODO: Implement a way to register custom metrics.
        base_class = ExampleMetric if self._src.__class__.__name__ == "Example" else Metric
        return {m.attr_name.lower(): m for m in base_class.__subclasses__()}

    @property
    def metrics(self) -> list[Metric]:
        """Get the list of available metrics."""
        return [metric_name for metric_name in self.available_metrics.keys() if metric_name in self.__dict__]

    def __getattr__(self, name):
        """Get a metric by name.

        If the metric is not already computed, it will be created and set as an attribute of the MetricCollection.
        """
        if name in self.available_metrics:
            metric_instance = self.available_metrics[name](self._src)
            setattr(self, name, metric_instance)
        else:
            raise AttributeError(f"Metric '{name}' not found in available metrics.")

        return metric_instance

    def __repr__(self):
        return f"MetricCollection({self.metrics})"
