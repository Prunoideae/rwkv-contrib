"""
debug_tools
===========

This module contains tools for debugging and testing.
"""


from copy import deepcopy
from numpy import ndarray
import numpy as np
from torch import Tensor
from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from rwkv_contrib.pipeline import StatefulPipeline, GenerationArgs


class StateDump:
    """
    Represents a dump of all 32 layers in the state of RWKV-LM.
    """

    layers: list[list[float]]
    labels: list[str]
    output_prefix: str
    token_count: int

    def __init__(self, output_prefix: str) -> None:
        self.layers = []
        self.labels = []
        self.output_prefix = output_prefix
        self.token_count = 0

    def wraps(self, pipeline: "StatefulPipeline") -> None:
        """
        Wraps the 'infer' method of a pipeline.
        """

        pipeline_infer = pipeline.infer

        def infer(tokens: list[int], args: "GenerationArgs" = None) -> tuple[Optional[int], list[Tensor]]:
            """
            Wraps the 'infer' method of a pipeline.
            """
            token, state = pipeline_infer(tokens, args)
            state: list[Tensor]
            state_chunks: list[tuple[Tensor, ...]] = [chunk for chunk in zip(*[iter(deepcopy(state))] * 5)]
            for idx, (att_xx, att_aa, att_bb, att_pp, ffn_xx) in enumerate(state_chunks):
                # No att_pp here due to it's increasing, and will cause the umap repr to present some weird linear pattern.
                concated = att_aa.float().tolist() + att_bb.float().tolist() + att_xx.float().tolist() + ffn_xx.float().tolist()
                self.layers.append(concated)
                self.labels.append(f"{self.token_count}-l-{idx}")
            self.token_count += 1
            return token, state

        pipeline.infer = infer

    def dumps(self) -> None:
        """
        Dumps the state to disk.

        Will clear the state after dumping.
        """
        layers_output = Path(f"{self.output_prefix}.layers.npy")
        labels_output = Path(f"{self.output_prefix}.labels.npy")

        np.save(layers_output, np.array(self.layers))
        np.save(labels_output, np.array(self.labels))
 

        self.layers = []
        self.labels = []
        self.token_count = 0

    def loads(self) -> tuple[ndarray, ndarray]:
        """
        Loads the state from disk.

        Returns a tuple of (layers, labels).
        """

        layers_output = Path(f"{self.output_prefix}.layers.npy")
        labels_output = Path(f"{self.output_prefix}.labels.npy")
  
        return np.load(layers_output), np.load(labels_output)
