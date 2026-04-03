from __future__ import annotations
from typing import List

# this is an iterator this is sort the items(prompt,response)
# this will reset its index when iterating finished or itrerating again


class LengthCurriculum:
    def __init__(self, items: List[tuple[str, str]]):
        self.items = sorted(items, key=lambda b: len(b[0]))
        self._i = 0

    def __iter__(
        self,
    ):
        self._i = 0
        return self

    def __next__(
        self,
    ):
        if self._i >= len(self.items):
            raise StopIteration
        it = self.items[self._i]
        self._i += 1
        return it
