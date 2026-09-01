__all__ = [
    'Statistician',
]

import torch


class Statistician:

    def __init__(self, chunk_size: int) -> None:
        self._chunk_size = chunk_size
        self._means: list[torch.Tensor] = []
        self._variances: list[torch.Tensor] = []
        self._running_chunk: list[torch.Tensor] = []

    @property
    def _running_chunk_size(self) -> int:
        return sum(samples.shape[0] for samples in self._running_chunk)

    @property
    def num_samples(self) -> int:
        num_chunks, = {len(self._means), len(self._variances)}  # noqa: E501 pylint: disable=unbalanced-tuple-unpacking
        return num_chunks * self._chunk_size + self._running_chunk_size

    def update(self, samples: torch.Tensor) -> None:
        assert samples.dim() == 2
        self._running_chunk.append(samples)

        if self._running_chunk_size < self._chunk_size:
            return

        running_chunk = torch.cat(self._running_chunk)
        while running_chunk.shape[0] >= self._chunk_size:
            chunk = running_chunk[:self._chunk_size]
            self._means.append(chunk.mean(0))
            self._variances.append(chunk.var(0))
            running_chunk = running_chunk[self._chunk_size:]

        if running_chunk.shape[0]:
            self._running_chunk = [running_chunk]

    def _weighted_average(
        self,
        value: torch.Tensor | None,
        running_value: torch.Tensor | None,
    ) -> torch.Tensor:
        if value is None and running_value is None:
            raise RuntimeError("No samples to compute")
        if value is None:
            assert running_value is not None
            return running_value
        if running_value is None:
            assert value is not None
            return value

        w = self._running_chunk_size / self.num_samples
        return (1 - w) * value + w * running_value

    def compute_mean(self) -> torch.Tensor:
        mean = torch.stack(self._means).mean(0) if self._means else None
        running_mean = (
            torch.cat(self._running_chunk).mean(0)
            if self._running_chunk else None
        )
        return self._weighted_average(mean, running_mean)

    def compute_variance(
        self,
        mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mean is None:
            mean = self.compute_mean()

        if self._variances:
            variances = [
                chunk_variance + (chunk_mean - mean)**2 for chunk_mean,
                chunk_variance in zip(self._means, self._variances)
            ]
            variance = torch.stack(variances).mean(0)
        else:
            variance = None

        if self._running_chunk:
            running_chunk = torch.cat(self._running_chunk)
            squared_deviation = (running_chunk - mean)**2
            running_variance = squared_deviation.mean(0)
        else:
            running_variance = None

        return self._weighted_average(variance, running_variance)
