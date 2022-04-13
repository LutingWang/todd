import einops
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Union

import torch


T = TypeVar('T')

class BaseAccuracy(Generic[T]):
    def __init__(self, *args, n: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._corrects = np.zeros(n, dtype=int)
        self._total = 0
    
    def _accumulate(self, corrects: np.ndarray, total: int):
        self._corrects += corrects
        self._total += total

    def _todict(self, corrects: np.ndarray, total: Union[int, np.ndarray]) -> np.ndarray:
        corrects = corrects.astype(float)
        with np.errstate(invalid='ignore'):
            return corrects * 100 / total

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> T:
        pass
        
    def todict(self) -> T:
        return self._todict(self._corrects, self._total)


class Accuracy(BaseAccuracy[Dict[int, float]]):
    def __init__(self, topks: Tuple[int] = (1, 5)):
        super().__init__(n=len(topks))
        self._topks = topks

    def evaluate(self, preds: torch.Tensor, targets: torch.Tensor, accumulate: bool = True) -> Dict[int, float]:
        """
        Args:
            preds: m x k
            targets: m

        Returns:
            accuracies: n
                Accuracy for each `topk`.
        """
        m, k = preds.shape
        assert (targets >= 0).all() and (targets < k).all()

        maxk = max(self._topks)
        preds = preds.topk(maxk).indices
        targets = einops.repeat(targets, 'm -> m maxk', maxk=maxk)
        corrects = preds == targets
        corrects = np.array([corrects[:, :topk].sum().item() for topk in self._topks], dtype=int)
        if not accumulate:
            return self._todict(corrects, m)
        self._accumulate(corrects, m)
        return self.todict()

    def _todict(self, corrects: np.ndarray, total: int) -> Dict[int, float]:
        accuracies = super()._todict(corrects, total).tolist()
        return dict(zip(self._topks, accuracies))

    def __repr__(self) -> str:
        accuracies = self.todict()
        df = pd.DataFrame(dict(topk=accuracies.keys(), accuracies=accuracies.values()))
        return repr(df)


class BinaryAccuracy(BaseAccuracy[Dict[float, float]]):
    def __init__(self, thrs: Tuple[float] = (0.1, 0.5, 0.9)):
        super().__init__(n=len(thrs))
        self._thrs = thrs
        self._tps = self._corrects.copy()
        self._fps = self._corrects.copy()

    def evaluate(self, preds: torch.Tensor, targets: torch.Tensor, accumulate: bool = True) -> List[float]:
        """
        Args:
            preds: m x k | m
            targets: m

        Returns:
            accuracies: n
                Accuracy for each `thr`.
        """
        if preds.ndim == 2:
            preds = preds.max(-1).values
        assert preds.ndim == 1
        m, = preds.shape

        if targets.dtype != torch.bool:
            targets = targets >= 0

        preds = [preds >= thr for thr in self._thrs]
        corrects = np.array([(p == targets).sum() for p in preds], dtype=int)
        tps = np.array([(p & targets).sum() for p in preds], dtype=int)
        fps = np.array([(p & ~targets).sum() for p in preds], dtype=int)
        if not accumulate:
            return self._todict(corrects, tps, fps, m)
        self._accumulate(corrects, tps, fps, m)
        return self.todict()

    def _accumulate(self, corrects: np.ndarray, tps: np.ndarray, fps: np.ndarray, total: int):
        super()._accumulate(corrects, total)
        self._tps += tps
        self._fps += fps

    def _todict(self, corrects: np.ndarray, tps: np.ndarray, fps: np.ndarray, total: int) -> Dict[str, Dict[float, float]]:
        tns = corrects - tps
        fns = total - corrects - fps
        results = dict(
            accuracies=super()._todict(corrects, total),
            recalls=super()._todict(tps, tps + fns),
            precisions=super()._todict(tps, tps + fps),
            fprs=super()._todict(fps, tns + fps),
        )
        return {k: dict(zip(self._thrs, v)) for k, v in results.items()}

    def todict(self) -> Dict[str, Dict[float, float]]:
        return self._todict(self._corrects, self._tps, self._fps, self._total)

    def __repr__(self) -> str:
        d = dict(thr=self._thrs)
        d.update({k: v.values() for k, v in self.todict().items()})
        df = pd.DataFrame(d)
        return repr(df)
