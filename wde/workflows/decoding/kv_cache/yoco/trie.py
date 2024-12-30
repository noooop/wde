from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Node:
    child: Dict[int, "Node"] = field(default_factory=dict)
    items: Dict[int, Any] = field(default_factory=dict)


class Trie:

    def __init__(self):
        self._root: Node = Node()

    def find(self, delta_token_ids: List[int]):
        """
        return
        1. empty list: not candidate
        2. Match tokens longer than prefix: hit computed
        3. Match tokens shorter than prefix: need compute
        """
        node = self._root

        hit = 0
        for t in delta_token_ids:
            if t not in node.child:
                break
            else:
                node = node.child[t]
                hit += 1

        return hit, list(node.items.values())

    def insert(self, delta_token_ids: List[int], item):
        item_id = id(item)

        node = self._root
        for t in delta_token_ids:
            if t not in node.child:
                node.child[t] = Node()

            node = node.child[t]
            node.items[item_id] = item

    def delete(self, delta_token_ids: List[int], item):
        item_id = id(item)

        node = self._root
        for t in delta_token_ids:
            if t not in node.child:
                break

            tmp_node = node.child[t]

            del tmp_node.items[item_id]

            if len(tmp_node.items) == 0:
                del node.child[t]
                break

            else:
                node = tmp_node
