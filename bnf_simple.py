from functools import cache
from rwkv_contrib.bnf import Action, BNFTree, Node, token_size, token_table


class SimpleNode(Node):
    def get_sub(self, idx: int | None) -> int | None:
        return None

    def handle_match(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()  # pop current node as we are done with it
        return Action.CONSUME

    def handle_fail(self, callstack: list[tuple[int, int]]) -> Action:
        callstack.pop()
        return Action.FAIL

    def add_to_stack(self, callstack: list[tuple[int, int]]) -> None:
        callstack.append((self.bnf_index, 0))

    def matched(self, callstack: list[tuple[int, int]]) -> bool:
        return False  # if simple node is on the stack, it is not matched yet or it would have been popped


class CharNode(SimpleNode):
    """
    Represents a node that accepts a single character.
    """

    char: bytes

    def __init__(self, bnf_tree: BNFTree, char: bytes) -> None:
        self.char = char
        super().__init__(bnf_tree)

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return self.char == char

    @cache
    def get_logits(self) -> set[int]:
        return {k for k, v in token_table.items() if v.startswith(self.char)}


class NotCharNode(SimpleNode):
    """
    Represents a node that accepts any character except a single character.
    """

    char: bytes

    def __init__(self, bnf_tree: BNFTree, char: bytes) -> None:
        self.char = char
        super().__init__(bnf_tree)

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return self.char != char

    @cache
    def get_logits(self) -> set[int]:
        return {k for k, v in token_table.items() if not v.startswith(self.char)}


class WildcardNode(SimpleNode):
    """
    Represents a node that accepts any character.
    """

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return char is not None  # any character is accepted but None (end of string) is not

    @cache
    def get_logits(self) -> set[int]:
        return {x for x in token_table.items()}


class DigitNode(SimpleNode):
    """
    Represents a node that accepts any digit.
    """

    def accept_character(self, char: bytes | None, callstack: list[tuple[int, int]]) -> bool:
        return char is not None and char.isdigit()  # any digit is accepted but None (end of string) is not

    @cache
    def get_logits(self) -> set[int]:
        return {x for x in range(token_size) if x in token_table and token_table[x].isdigit()}
