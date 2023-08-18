from functools import cache
from rwkv_contrib.bnf import BNFTree, Node
from rwkv_contrib.bnf_complex import OrNode, PopNode, RepeatNode, SequenceNode
from rwkv_contrib.bnf_simple import CharNode, NotCharNode

"""
A toolset for Markdown-like BNF grammars.
"""


@cache
def literal(tree: BNFTree, literal: bytes):
    literal_parsed = [bytes([x]) for x in literal]
    return SequenceNode(tree, [CharNode(tree, x) for x in literal_parsed])


@cache
def non_newline(tree: BNFTree) -> Node:
    return RepeatNode(tree, NotCharNode(tree, b"\n"))


@cache
def quoted(tree: BNFTree, quote: bytes, inner: Node = None) -> Node:
    if inner is None:
        inner = NotCharNode(tree, b"\n")
    quote = [bytes([x]) for x in quote]
    start, end = quote
    return SequenceNode(
        tree,
        [
            CharNode(tree, start),
            RepeatNode(
                tree,
                OrNode(
                    tree,
                    [
                        PopNode(tree, CharNode(tree, end), depth=2),
                        inner,
                    ],
                ),
            ),
        ],
    )


@cache
def asterisks(tree: BNFTree, inner: Node = None) -> Node:
    return quoted(tree, b"**", inner)


@cache
def parentheses(tree: BNFTree, inner: Node = None) -> Node:
    return quoted(tree, b"()", inner)


@cache
def square_brackets(tree: BNFTree, inner: Node = None) -> Node:
    return quoted(tree, b"[]", inner)


@cache
def curly_brackets(tree: BNFTree, inner: Node = None) -> Node:
    return quoted(tree, b"{}", inner)


@cache
def infinite_list(tree: BNFTree, bullet: bytes, item_constraint: Node = None) -> Node:
    """
    Represents an unnumbered list of items.

    Such list is defined as:

    ```
    {bullet} {item_constraint}{newline}...
    ```

    Note that this is infinite.
    """

    if item_constraint is None:
        item_constraint = non_newline(tree)

    return RepeatNode(
        tree,
        SequenceNode(
            tree,
            [
                CharNode(tree, bullet),
                CharNode(tree, b" "),
                item_constraint,
                CharNode(tree, b"\n"),
            ],
        ),
    )


@cache
def finite_list(tree: BNFTree, bullet: bytes, item_count: int, item_constraint: Node = None) -> Node:
    if item_constraint is None:
        item_constraint = non_newline(tree)
    single_item = SequenceNode(
        tree,
        [
            CharNode(tree, bullet),
            CharNode(tree, b" "),
            item_constraint,
            CharNode(tree, b"\n"),
        ],
    )
    return SequenceNode(tree, [single_item] * item_count)


@cache
def numbered_list(tree: BNFTree, item_count: int, item_constraint: Node = None) -> Node:
    if item_constraint is None:
        item_constraint = non_newline(tree)

    number_items = []
    for i in range(1, item_count + 1):
        number_items.append(
            SequenceNode(
                tree,
                [
                    CharNode(tree, str(i).encode()),
                    CharNode(tree, b"."),
                    CharNode(tree, b" "),
                    item_constraint,
                    CharNode(tree, b"\n"),
                ],
            )
        )

    return SequenceNode(tree, number_items)


def named_list(tree: BNFTree, bullet: bytes | None, nodes: dict[str, Node | None]):
    node_sequence = []
    for key, value in nodes.items():
        node_sequence.append(literal(tree, (bullet + b" " if bullet is not None else b"") + key.encode() + b": "))
        if value is None:
            value = non_newline(tree)
        node_sequence.append(value)
        node_sequence.append(CharNode(tree, b"\n"))

    return SequenceNode(tree, node_sequence)


@cache
def choices(tree, *args: str):
    args: list[list[bytes]] = [[bytes([y]) for y in x.encode()] for x in args]
    # make trie of all strings
    trie = {}
    for arg in args:
        current = trie
        for char in arg:
            if char not in current:
                current[char] = {}
            current = current[char]

    def trie_to_node(trie):
        if trie == {}:
            raise ValueError("Empty trie")
        return OrNode(
            tree,
            [
                SequenceNode(
                    tree,
                    [CharNode(tree, x), trie_to_node(trie[x])],
                )
                if trie[x] != {}
                else CharNode(tree, x)
                for x in trie
            ],
        )

    return trie_to_node(trie)


def list_line(tree, item_constraint: Node = None):
    if item_constraint is None:
        item_constraint = non_newline(tree)
    return SequenceNode(
        tree,
        [
            item_constraint,
            RepeatNode(
                tree,
                SequenceNode(
                    tree,
                    [
                        CharNode(tree, b","),
                        CharNode(tree, b" "),
                        item_constraint,
                    ],
                ),
            ),
        ],
    )
