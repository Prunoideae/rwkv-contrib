from functools import cache, lru_cache
from typing import Any
import ujson

from rwkv_contrib.bnf import BNFTree, Node
from rwkv_contrib.bnf_complex import OptionalNode, SequenceNode, OrNode, RepeatNode, PopNode
from rwkv_contrib.bnf_simple import CharNode, DigitNode, NotCharNode, WildcardNode


@cache
def json_string(tree: BNFTree):
    return SequenceNode(
        tree,
        [
            CharNode(tree, b'"'),
            RepeatNode(
                tree,
                OrNode(
                    tree,
                    [
                        SequenceNode(tree, [CharNode(tree, b"\\"), NotCharNode(tree, b"\n")]),
                        PopNode(tree, CharNode(tree, b'"'), depth=2),
                        NotCharNode(tree, b"\n"),
                    ],
                ),
            ),
        ],
    )


@cache
def literal(tree: BNFTree, literal: bytes):
    literal_parsed = [bytes([x]) for x in literal]
    return SequenceNode(tree, [CharNode(tree, x) for x in literal_parsed])


@cache
def json_string_literal(tree: BNFTree, literal_string: bytes):
    jsonified = ujson.dumps(literal_string, reject_bytes=False).encode()
    return literal(tree, jsonified)


@cache
def json_string_enum(tree, *args: str):
    args = [ujson.dumps(x) for x in args]
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


@cache
def json_boolean(tree: BNFTree):
    return OrNode(tree, [literal(tree, b"true"), literal(tree, b"false")])


@cache
def json_number(tree):
    return SequenceNode(
        tree,
        [
            OptionalNode(
                tree,
                CharNode(tree, b"-"),
            ),
            RepeatNode(
                tree,
                OrNode(
                    tree,
                    [
                        DigitNode(tree),
                        PopNode(
                            tree,
                            CharNode(tree, b"."),
                            depth=2,
                        ),
                    ],
                ),
            ),
            RepeatNode(
                tree,
                DigitNode(tree),
            ),
        ],
    )


@cache
def json_date(tree):
    digit_match = DigitNode(tree)
    return SequenceNode(
        tree,
        [
            CharNode(tree, b'"'),
            digit_match,
            digit_match,
            digit_match,
            digit_match,
            CharNode(tree, b"-"),
            digit_match,
            digit_match,
            CharNode(tree, b"-"),
            digit_match,
            digit_match,
            CharNode(tree, b'"'),
        ],
    )


@cache
def json_time(tree):
    digit_match = DigitNode(tree)
    return SequenceNode(
        tree,
        [
            CharNode(tree, b'"'),
            digit_match,
            digit_match,
            CharNode(tree, b":"),
            digit_match,
            digit_match,
            CharNode(tree, b":"),
            digit_match,
            digit_match,
            CharNode(tree, b'"'),
        ],
    )


@cache
def json_time(tree):
    return SequenceNode(
        tree,
        [
            CharNode(tree, b'"'),
            DigitNode(tree),
            DigitNode(tree),
            CharNode(tree, b":"),
            DigitNode(tree),
            DigitNode(tree),
            CharNode(tree, b":"),
            DigitNode(tree),
            DigitNode(tree),
            CharNode(tree, b'"'),
        ],
    )


@cache
def json_null(tree):
    return literal(tree, b"null")


@cache
def json_nullable(tree, node: Node):
    return OrNode(tree, [json_null(tree), node])


@cache
def json_array(tree, node: Node):
    return SequenceNode(
        tree,
        [
            CharNode(tree, b"["),
            OrNode(
                tree,
                [
                    CharNode(tree, b"]"),
                    SequenceNode(
                        tree,
                        [
                            node,
                            RepeatNode(
                                tree,
                                SequenceNode(
                                    tree,
                                    [
                                        literal(tree, b", "),
                                        node,
                                    ],
                                ),
                            ),
                            CharNode(tree, b"]"),
                        ],
                    ),
                ],
            ),
        ],
    )


def json_object(tree, nodes: dict[str, Node]):
    node_sequence = []
    for key, value in nodes.items():
        node_sequence.append(json_string_literal(tree, key.encode()))
        node_sequence.append(literal(tree, b": "))
        node_sequence.append(value)
        node_sequence.append(literal(tree, b", "))
    node_sequence.pop()

    return SequenceNode(
        tree,
        [
            CharNode(tree, b"{"),
            SequenceNode(
                tree,
                node_sequence,
            ),
            CharNode(tree, b"}"),
        ],
    )


def load_from_schema(tree: BNFTree, json: dict[str, Any]):
    element_type = json["type"]
    if element_type == "string":
        return json_string(tree)
    elif element_type == "number":
        return json_number(tree)
    elif element_type == "boolean":
        return json_boolean(tree)
    elif element_type == "array":
        return json_array(tree, load_from_schema(tree, json["items"]))
    elif element_type == "object":
        return json_object(
            tree,
            {key: load_from_schema(tree, value) for key, value in json["properties"].items()},
        )
    elif element_type == "enum":
        return json_string_enum(tree, *json["values"])
    elif element_type == "date":
        return json_date(tree)
    elif element_type == "time":
        return json_time(tree)
    elif element_type == "nullable":
        return json_nullable(tree, load_from_schema(tree, json["value"]))
