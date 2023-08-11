from functools import cache
from rwkv_contrib.bnf import BNFTree, Node, Action


class SequenceNode(Node):
    nodes: list[Node]

    def __init__(self, bnf_tree: BNFTree, nodes: list[Node]) -> None:
        self.nodes = nodes
        super().__init__(bnf_tree)

    def add_to_stack(self, callstack: list[tuple[int, int]]) -> None:
        callstack.append((self.bnf_index, 0))
        self.bnf_tree[self.get_sub(None)].add_to_stack(callstack)

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return True

    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        # last item in the callstack is always current sequence node
        cur_idx, cur_sub = callstack[-1]
        next_sub = cur_sub + 1
        next_sub_idx = self.get_sub(next_sub)
        if next_sub_idx is None:
            # we are at the end of the sequence node, so we need to pop current node
            # and leave the character for parent node
            callstack.pop()
        else:
            # we are not at the end of the sequence node, so we step to the next subnode
            # and hand the character to it
            callstack[-1] = (cur_idx, next_sub)
            self.bnf_tree[next_sub_idx].add_to_stack(callstack)
        # in both cases we need to re-evaluate the current node
        return Action.RE_EVAL

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        return Action.FAIL

    def get_sub(self, idx: int | None) -> int | None:
        if idx is None:
            return self.nodes[0].bnf_index
        elif idx < len(self.nodes):
            return self.nodes[idx].bnf_index
        else:
            return None

    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        cur_idx, cur_sub = callstack[-1]
        return cur_sub == len(self.nodes) - 1  # we are already at the end of the sequence node

    @cache
    def get_logits(self) -> set[int]:
        logits = set()
        for node in self.nodes:
            logits.update(node.get_logits())
        return logits


class OrNode(SequenceNode):
    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()  # pop current node as we are done with it
        return Action.RE_EVAL

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        cur_idx, cur_sub = callstack[-1]
        next_sub_idx = self.get_sub(cur_sub + 1)
        if next_sub_idx is None:  # we tried all subnodes and failed, so we need to pop current node and fail
            callstack.pop()
            return Action.FAIL
        else:
            # we are not at the end of the sequence node, so we step to the next subnode to try it
            callstack[-1] = (cur_idx, cur_sub + 1)
            self.bnf_tree[next_sub_idx].add_to_stack(callstack)
            return Action.RE_EVAL

    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        return True  # if the node is on top of the stack, it is matched since the matched subnode is popped

    @cache
    def get_logits(self) -> set[int]:
        logits = set()
        for node in self.nodes:
            logits |= node.get_logits()
        return logits


class PopNode(Node):
    depth: int
    node: Node

    def __init__(self, bnf_tree: BNFTree, node: Node, depth=1) -> None:
        self.depth = depth
        self.node = node
        super().__init__(bnf_tree)

    def add_to_stack(self, callstack: list[tuple[int, int]]) -> None:
        callstack.append((self.bnf_index, 0))
        self.node.add_to_stack(callstack)

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return True

    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        depth = self.depth
        while depth > 0:
            callstack.pop()
            depth -= 1
        return Action.RE_EVAL

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        return Action.FAIL

    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        depth = self.depth
        while depth > 0:
            callstack.pop()
            depth -= 1
        return True  # it should not be possible to match a pop node

    @cache
    def get_logits(self) -> set[int]:
        return self.node.get_logits()


class RepeatNode(Node):
    node: Node

    def __init__(self, bnf_tree: BNFTree, node: Node) -> None:
        self.node = node
        super().__init__(bnf_tree)

    def add_to_stack(self, callstack: list[tuple[int, int]]) -> None:
        callstack.append((self.bnf_index, 0))
        self.node.add_to_stack(callstack)

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return True

    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        self.node.add_to_stack(callstack)
        return Action.RE_EVAL

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        return Action.RE_EVAL

    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        return False  # it should not be possible to match a repeat node

    @cache
    def get_logits(self) -> set[int]:
        return self.node.get_logits()


class OptionalNode(Node):
    node: Node

    def __init__(self, bnf_tree: BNFTree, node: Node) -> None:
        self.node = node
        super().__init__(bnf_tree)

    def add_to_stack(self, callstack: list[tuple[int, int]]) -> None:
        callstack.append((self.bnf_index, 0))
        self.node.add_to_stack(callstack)

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return True

    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        return Action.RE_EVAL

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        return Action.RE_EVAL

    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        return True  # it should not be possible to match a optional node

    @cache
    def get_logits(self) -> set[int]:
        return self.node.get_logits()
