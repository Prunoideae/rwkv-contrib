"""
bnf.py

This module contains the classes and functions for parsing BNF grammars for RWKV-LM.
"""
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import cache
from typing import Any, Optional
import numpy as np
from pytrie import SortedStringTrie

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import torch

base_tokenizer: TRIE_TOKENIZER = None
token_size = 65536
impossible = -1e9
token_table: dict[int, bytes] = {}
tensor_dtype = torch.float16


def setup_tokens():
    for i in range(token_size):
        try:
            token_table[i] = base_tokenizer.decodeBytes([i])
        except:
            pass


class BNFException(Exception):
    ...


class Action(Enum):
    """
    Represents an action in the automata.
    """

    RE_EVAL = 0  # Re-evaluate the current node.
    CONSUME = 1  # Consumes current character.
    FAIL = 2  # Tell the parent node that it failed.


class BNFTree:
    nodes: list["Node"]

    logits_cache: dict[tuple[tuple[int, int]], np.ndarray[Any, np.dtype[bool]]]

    def __init__(self) -> None:
        self.nodes = []
        self.logits_cache = {}

    def __getitem__(self, idx: int) -> "Node":
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def register(self, node: "Node") -> int:
        """
        Registers a node to the tree. Returns the index of the node.
        """
        self.nodes.append(node)
        return len(self.nodes) - 1

    def eval_once(self, char: bytes | None, callstack: list[tuple[int, int]], propagate_fail=False) -> Action:
        cur_idx, cur_sub = callstack[-1]
        node = self[cur_idx]
        if node.accept_character(char, callstack) and not propagate_fail:
            action = node.handle_match(callstack)
        else:
            action = node.handle_fail(callstack)
        return action

    def eval(self, char: bytes | None, callstack: list[tuple[int, int]]) -> None:
        action = self.eval_once(char, callstack)
        if callstack == []:
            raise BNFException("Callstack is empty.")
        while action != Action.CONSUME:
            action = self.eval_once(char, callstack, propagate_fail=action == Action.FAIL)
            if callstack == []:
                raise BNFException("Callstack is empty.")

    def eval_bytes(self, text: bytes, callstack: list[tuple[int, int]]) -> None:
        for char in text:
            self.eval(bytes([char]), callstack)

    def eval_token(self, token: int, callstack: list[tuple[int, int]]) -> None:
        token_bytes = token_table[token]
        self.eval_bytes(token_bytes, callstack)

    def deflate(self, callstack: list[tuple[int, int]]) -> bool:
        """
        Deflates the callstack. Returns True if the callstack is empty (all resolved).

        It will remove all nodes that are resolved from the callstack.
        """
        if not callstack:
            return True
        while callstack:
            cur_idx, cur_sub = callstack[-1]
            node = self[cur_idx]
            if not node.matched(callstack):
                return False
            else:
                callstack.pop()  # pop current node as we are done with it
        return True

    def get_logits(self, callstack: list[tuple[int, int]]) -> np.ndarray[Any, np.dtype[bool]]:
        """
        Get the logits for the current callstack.
        """

        callstack = list(callstack)
        if tuple(callstack) not in self.logits_cache:
            logits_filter = np.zeros(token_size, dtype=bool)
            for logit, logit_bytes in token_table.items():
                try:
                    cur_callstack = deepcopy(callstack)
                    self.eval_bytes(logit_bytes, cur_callstack)
                    logits_filter[logit] = True
                except BNFException as e:
                    pass
            self.logits_cache[tuple(callstack)] = logits_filter
        return self.logits_cache[tuple(callstack)]

    def dump_logits(self, path) -> None:
        """
        Dump the logits cache to a file.
        """
        import numpy as np

        ks: list[np.ndarray] = []
        vs = []
        for k, v in self.logits_cache.items():
            k = np.array(k).flatten()
            ks.append(k)
            vs.append(v)

        max_len = max([len(k) for k in ks])
        for k in ks:
            original_len = len(k)
            k.resize(max_len, refcheck=False)
            k[original_len:] = -1
        np.savez_compressed(path, ks=ks, vs=vs)

    def load_logits(self, path) -> None:
        """
        Load the logits cache from a file.
        """
        import numpy as np

        data = np.load(path)
        ks = data["ks"]
        vs = data["vs"]

        for k, v in zip(ks, vs):
            kt = list()
            for i in range(0, len(k), 2):
                if k[i] == -1:
                    break
                kt.append((k[i], k[i + 1]))
            self.logits_cache[tuple(kt)] = v

    @cache
    def get_tensor(self, device: str, callstack: tuple[tuple[int, int]]) -> torch.Tensor:
        """
        Get the filter tensor for the current callstack.
        """
        logits = self.get_logits(callstack)
        # set false to impossible, true to 0
        logits = np.where(logits, 0, impossible)
        return torch.tensor(logits, dtype=tensor_dtype, device=device).float()

    def filter_tensor(self, tensor: torch.Tensor, callstack: list[tuple[int, int]]) -> torch.Tensor:
        """
        Filter the tensor by the callstack.
        """
        filter_tensor = self.get_tensor(tensor.device, tuple(callstack))
        return tensor + filter_tensor


class Node:
    """
    Represents a node in the automata.
    """

    bnf_tree: BNFTree
    bnf_index: int

    def __init__(self, bnf_tree: BNFTree) -> None:
        self.bnf_tree = bnf_tree
        self.bnf_index = bnf_tree.register(self)
        self.get_logits()  # cache the logits

    @abstractmethod
    def add_to_stack(self, callstack: list[tuple[int, int]]) -> None:
        """
        Adds the node to the stack.
        """

    @abstractmethod
    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        """
        Accepts a character.
        """

    @abstractmethod
    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        """
        Get the action to take after the character is accepted.
        """

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        """
        Get the action to take after the character is rejected.
        """

    @abstractmethod
    def get_sub(self, idx: Optional[int]) -> Optional[int]:
        """
        Returns the subnode at the given sub index.

        If the sub index is None, returns the first subnode.
        """

    @abstractmethod
    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        """
        Check if the node is matched.
        """

    @abstractmethod
    def get_logits(self) -> set[int]:
        """
        Get the logits for the current node.
        """

    @property
    @cache
    def complex(self) -> bool:
        return self.get_sub(None) is not None
