from collections import deque
from typing import TYPE_CHECKING, Any, Generator, Optional

import numpy as np
import torch
from torch import Tensor
from os import path
from rwkv_contrib.bnf import BNFTree, Node
from rwkv_contrib.penalty import GlobalPenalty, Penalty
from rwkv_contrib.pipeline import GenerationArgs
from rwkv_contrib.tokenizer import RWKVTokenizer, Tokenizer

import torch.nn.functional as F

if TYPE_CHECKING:
    from rwkv.model import RWKV


class BNFPipeline:
    """
    A stateless pipeline for RWKV.

    Output is restricted by a BNF grammar.
    """

    def __init__(
        self,
        model: "RWKV",
        tree: BNFTree,
        initial_node: Node,
        tokenizer: Tokenizer[str] = RWKVTokenizer(),
        penalty: Penalty = None,
        default_args: GenerationArgs = None,
        logits_cache: str = None,
    ) -> None:
        penalty = penalty or GlobalPenalty()
        default_args = default_args or GenerationArgs()

        self.model = model
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.default_args = default_args

        self.encode = tokenizer.encode
        self.decode = tokenizer.decode

        self.tree = tree
        self.initial_node = initial_node
        self.callstack: list[tuple[int, int]] = []

        self.logits_cache = logits_cache
        if logits_cache is not None and path.exists(logits_cache):
            self.tree.load_logits(logits_cache)

    def sample_logits(self, logits: Tensor, args: GenerationArgs) -> Tensor:
        """
        Sample logits.
        """
        args = args or self.default_args

        probs = F.softmax(logits, dim=-1)
        top_k = args.top_k
        if probs.device == torch.device("cpu"):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > args.top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if args.temperature != 1.0:
                probs = probs ** (1.0 / args.temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            token = int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > args.top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if args.temperature != 1.0:
                probs = probs ** (1.0 / args.temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            token = int(out)

        return token

    def dump_logits(self, logits_cache: str = None) -> None:
        logits_cache = logits_cache or self.logits_cache
        self.tree.dump_logits(logits_cache)

    def infer(
        self,
        tokens: list[int],
        *,
        state: Any = None,
        args: GenerationArgs = None,
        penalty: Penalty = None,
        update_tokens_penalty: bool = True,
        intialize_callstack: bool = True,
    ) -> tuple[Optional[int], Any]:
        """
        Infer the next token from a list of tokens.

        If the input is a list, and the first element is an integer, it is assumed to be a list of tokens.

        None is returned if stop tokens are generated.
        """

        args = args or self.default_args
        penalty = penalty or self.penalty

        if intialize_callstack:
            self.callstack = []
            self.initial_node.add_to_stack(self.callstack)

        if update_tokens_penalty:
            for token in tokens:
                penalty.update(token, args)

        for i in range(0, len(tokens), args.chunk_len):
            chunk = tokens[i : i + args.chunk_len]
            out, state = self.model.forward(chunk, state=state)

        for n in args.token_ban:
            out[n] = -float("inf")

        out = penalty.transform(out, args)
        out = self.tree.filter_tensor(out, self.callstack)

        token = self.sample_logits(out, args=args)
        self.tree.eval_token(token, self.callstack)
        if token in args.token_stop:
            return None, state

        return token, state

    def generate(self, ctx: str, generation_length: int = 100, *, state=None, args: GenerationArgs = None, clear_penalty: bool = True) -> Generator[str, None, None]:
        self.callstack = []
        self.initial_node.add_to_stack(self.callstack)

        if args is None:
            args = self.default_args

        if clear_penalty:
            self.penalty.clear()

        tokens_tmp = []
        token, state = self.infer(self.encode(ctx), state=state, args=args, intialize_callstack=False)
        while token is not None and generation_length > 0:
            generation_length -= 1
            tokens_tmp.append(token)
            tmp = self.decode(tokens_tmp)
            if self.tokenizer.validate(tmp):
                yield tmp
                tokens_tmp = []
            if self.tree.deflate(self.callstack):
                break
            token, state = self.infer([token], state=state, args=args, intialize_callstack=False)


class StatefulBNFPipeline(BNFPipeline):
    state: Any

    def __init__(
        self,
        model: "RWKV",
        tree: BNFTree,
        initial_node: Node,
        tokenizer: Tokenizer[str] = RWKVTokenizer(),
        penalty: Penalty = None,
        default_args: GenerationArgs = None,
        logits_cache: str = None,
        init_state: Any = None,
        init_prompt: str = None,
    ) -> None:
        super().__init__(model, tree, initial_node, tokenizer, penalty, default_args, logits_cache)

        self.state = init_state
        if init_prompt is not None:
            self.push(init_prompt)

    def infer(self, tokens: list[int], *, state: Any = None, args: GenerationArgs = None, penalty: Penalty = None) -> tuple[int | None, Any]:
        if state is None:
            state = self.state
            token, self.state = super().infer(tokens, state=state, args=args, penalty=penalty)
            return token, self.state
        return super().infer(tokens, state=state, args=args, penalty=penalty)

    def push(self, ctx: str):
        tokens = self.encode(ctx)
        _, self.state = self.infer(tokens, state=self.state, args=self.default_args, penalty=self.penalty)


class RecallableBNFPipeline(StatefulBNFPipeline):
    history: deque[Any]

    def __init__(
        self,
        model: "RWKV",
        tree: BNFTree,
        initial_node: Node,
        tokenizer: Tokenizer[str] = RWKVTokenizer(),
        penalty: Penalty = None,
        default_args: GenerationArgs = None,
        logits_cache: str = None,
        max_history: int = 10,
        init_state: Any = None,
        init_prompt: str = None,
    ) -> None:
        super().__init__(model, tree, initial_node, tokenizer, penalty, default_args, logits_cache, init_state, init_prompt)
        self.history = deque(maxlen=max_history)
        self.history.append(self.state)

    def recall(self, depth=1) -> Any:
        for _ in range(depth):
            self.state = self.history.pop()

    def push(self, ctx: str):
        self.history.append(self.state)
        return super().push(ctx)

    def generate(self, ctx: str, generation_length: int = 100, *, state=None, args: GenerationArgs = None, clear_penalty: bool = True) -> Generator[str, None, None]:
        self.history.append(self.state)
        return super().generate(ctx, generation_length, state=state, args=args, clear_penalty=clear_penalty)
