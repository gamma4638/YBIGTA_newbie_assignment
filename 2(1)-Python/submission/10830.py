from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod # self 없이 쓰는 정적메소드 - 인스턴스와 관련 없음 - 클래스 내부의 일반 함수
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # 구현하세요!
        # 행렬 인덱싱을 구현하는 매직 메소드
        # key: (행, 열) 인덱스를 받아 해당 위치에 value를 입력
        self.matrix[key[0]][key[1]] = value

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        # 구현하세요!
        # matmul을 활용해, n제곱을 제곱을 반복하는 분할 정복 형태로 구현
        # n제곱을 2로 나누거나, 1을 빼서 반복하는 과정으로 행렬 거듭제곱 구현
        result = Matrix.eye(self.shape[0])
        base = self.clone()  # 원본 행렬을 바꾸지 않기 위해 복제

        exp = n
        while exp > 0:
            if exp % 2 == 1:
                result = result @ base
            base = base @ base
            exp //= 2

        return result

    def __repr__(self) -> str:
        """
        행렬의 표현(디버깅 용도)를 문자열로 반환합니다.

        Returns:
            str: 행렬 정보 문자열.
        """
        return f"Matrix(shape={self.shape}, MOD={self.MOD}, matrix={self.matrix})"



from typing import Callable
import sys


"""
아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()