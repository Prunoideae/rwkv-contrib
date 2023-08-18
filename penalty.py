from copy import deepcopy
from typing import TYPE_CHECKING, Any, Deque, Dict
from abc import ABCMeta, abstractmethod
import numpy as np
from torch import Tensor
from collections import deque


if TYPE_CHECKING:
    from .pipeline import GenerationArgs


class Penalty(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        """
        Transform the logits with the penalty.
        """

    @abstractmethod
    def update(self, token: int, args: "GenerationArgs"):
        """
        Update the penalty with the token.
        """

    @abstractmethod
    def clear(self):
        """
        Clear the penalty.
        """

    @abstractmethod
    def copy(self) -> "Penalty":
        """
        Copy the penalty, for history.
        """


class GlobalPenalty(Penalty):
    def __init__(self) -> None:
        self.token_occurrences = {}

    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        for n in self.token_occurrences:
            out[n] -= args.alpha_presence + self.token_occurrences[n] * args.alpha_frequency
        return out

    def update(self, token: int, args: "GenerationArgs"):
        if token not in self.token_occurrences:
            self.token_occurrences[token] = 1
        else:
            self.token_occurrences[token] += 1

    def clear(self):
        self.token_occurrences = {}

    def copy(self) -> "GlobalPenalty":
        ret = GlobalPenalty()
        ret.token_occurrences = self.token_occurrences.copy()
        return ret


class SlidingPenalty(Penalty):
    def __init__(self, maxlen: int = 512) -> None:
        self.maxlen = maxlen
        self.token_occurrences: Deque[int] = deque()
        self.occurrences: Dict[int, int] = {}

    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        for n in self.occurrences:
            out[n] -= args.alpha_presence + self.occurrences[n] * args.alpha_frequency
        return out

    def update(self, token: int, args: "GenerationArgs"):
        self.token_occurrences.appendleft(token)
        if token not in self.occurrences:
            self.occurrences[token] = 1
        else:
            self.occurrences[token] += 1

        if len(self.token_occurrences) > self.maxlen:
            while len(self.token_occurrences) > self.maxlen:
                token = self.token_occurrences.pop()
                self.occurrences[token] -= 1

    def clear(self):
        self.token_occurrences.clear()
        self.occurrences = {}

    def copy(self) -> "SlidingPenalty":
        ret = SlidingPenalty(self.maxlen)
        ret.token_occurrences = self.token_occurrences.copy()
        ret.occurrences = self.occurrences.copy()
        return ret


class LogPenalty(Penalty):
    token_penalties: np.ndarray[Any, np.dtype[np.float32]]
    tensor_penalties: Tensor | None
    table_size: int

    def __init__(self, table_size: int = 65536) -> None:
        self.token_penalties = np.zeros(table_size, dtype=np.float32)
        self.table_size = table_size
        self.tensor_penalties = None

    def transform(self, out: Tensor, args: "GenerationArgs") -> Tensor:
        # create a tensor from penalties
        if self.tensor_penalties is None:
            penalties = Tensor(self.token_penalties)
            if out.device != penalties.device:
                penalties = penalties.to(out.device)
            self.tensor_penalties = penalties
        out -= self.tensor_penalties
        return out

    def update(self, token: int, args: "GenerationArgs"):
        to_update = self.token_penalties if self.tensor_penalties is None else self.tensor_penalties
        to_update *= args.alpha_decay
        to_update[token] += args.alpha_presence

    def clear(self):
        self.token_penalties = np.zeros(self.table_size, dtype=np.float32)
        self.tensor_penalties = None

    def copy(self) -> "LogPenalty":
        ret = LogPenalty(self.table_size)
        ret.token_penalties = deepcopy(self.token_penalties)
        ret.tensor_penalties = None if self.tensor_penalties is None else self.tensor_penalties.clone()
        return ret

