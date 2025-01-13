from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable
from math import factorial


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


# 3080 메모리 초과를 해결하지 못했습니다..


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: dict[T, int] = field(default_factory=dict) #인스턴스 별로 다른 값 할당
    is_end: bool = False


class Trie(list[TrieNode[T]]): # list[TrieNode[T]] 형태, TrieNode 상속
    def __init__(self) -> None:
        super().__init__() # 노드 클래스 init 호출
        self.append(TrieNode(body=None)) # 헤드 만들기

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        # 구현하세요!
        current_idx = 0
        
        current_node = self[current_idx]

        for char in seq:
            if char not in current_node.children:
                new_node = TrieNode(body=char)
                new_index = len(self)
                self.append(new_node)
                current_node.children[char] = new_index
            current_idx = current_node.children[char]

        self[current_idx].is_end = True


    def children_factorial_product(self) -> int:

        product: int = 1
        for node in self:
            count: int = len(node.children) + (1 if node.is_end else 0)
            product *= factorial(count)
        return product
