from typing import Any, Dict, Optional

from wde.workflows.decoding.kv_cache.logic_manager import NoFreeBlocksError


class Node:
    __slots__ = ("prev", "next", "key", "value")

    #  ↓ next <-    root   -> prev ↓
    # node <-> node <-> node <->  node
    #  ↑ head                 tail ↑

    def __init__(self,
                 prev: Optional["Node"] = None,
                 next: Optional["Node"] = None,
                 value: Any = None):
        self.prev = prev
        self.next = next
        self.value = value

    @classmethod
    def make_circle(cls):
        root = cls()
        root.prev = root
        root.next = root
        return root

    def reset(self):
        self.prev = None
        self.value = None

    def unlink(self):
        next = self.next
        prev = self.prev
        prev.next = next
        next.prev = prev

    def link_to_head(self, root):
        """
        root - next -> curr_head
           self ↑
        """
        self.next = root.next
        self.prev = root

        root.next.prev = self
        root.next = self

    def link_to_tail(self, root):
        """
        curr_tail <- prev - root
                 self ↑
        """

        self.next = root
        self.prev = root.prev

        root.prev.next = self
        root.prev = self

    def recycle(self, node: "Node"):
        # self is object_cache
        node.reset()
        node.next = self.next
        self.next = node.next

    def create(self):
        # self is object_cache
        if self.next is self:
            return Node()

        node = self.next
        self.next = node.next
        return node


class LRUEvictor:

    def __init__(self, node_class=None):
        assert issubclass(node_class, Node)
        self.node_class = node_class
        self.cache: Dict[int, node_class] = {}
        self.root = self.node_class.make_circle()

        # <= num_blocks, will not memory leaks
        self.object_cache = Node.make_circle()

    def __contains__(self, o) -> bool:
        return id(o) in self.cache

    def __len__(self):
        return len(self.cache)

    def evict(self):
        if len(self.cache) == 0:
            raise NoFreeBlocksError()

        tail = self.root.prev
        tail.unlink()

        value = tail.value
        key = id(value)

        del self.cache[key]
        self.object_cache.recycle(tail)
        return value

    def add(self, value):
        key = id(value)

        if key in self.cache:
            node = self.cache[key]
            node.unlink()
            node.value = value

        else:
            node = self.object_cache.create()
            node.value = value
            self.cache[key] = node

        node.link_to_head(self.root)

    def update(self, value):
        key = id(value)

        if key not in self.cache:
            return

        node = self.cache[key]
        node.unlink()
        node.link_to_head(self.root)
        return node

    def remove(self, value):
        key = id(value)

        if key not in self.cache:
            return

        node = self.cache[key]
        node.unlink()

        self.object_cache.recycle(node)
        del self.cache[key]
