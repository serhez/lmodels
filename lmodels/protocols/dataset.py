from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt


class DatasetSplit(Protocol):
    """An interface for a dataset split containing input and target pairs."""

    @property
    def inputs(self) -> npt.NDArray[Any]:
        """The inputs of the dataset split."""

        ...

    @property
    def targets(self) -> npt.NDArray[Any]:
        """The targets of the dataset split."""

        ...


@runtime_checkable
class Dataset(Protocol):
    """An interface for a dataset that can be split into training and test sets."""

    @property
    def training_set(self) -> DatasetSplit:
        """The training set of the dataset."""

        ...

    @property
    def test_set(self) -> DatasetSplit:
        """The test set of the dataset."""

        ...
