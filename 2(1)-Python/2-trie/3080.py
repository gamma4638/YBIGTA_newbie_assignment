from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    # 1. 문자열 개수 입력받기
    n = int(sys.stdin.readline().strip())

    # 2. Trie 초기화
    #    만약 문자(char)를 그대로 저장한다면 Trie[str] 형태로 선언해도 되고,
    #    문자 코드를 저장하려면 Trie[int] + push 시 ord(c)를 사용 가능.
    trie = Trie[int]()

    # 3. 문자열들을 트라이에 삽입
    for _ in range(n):
        s = sys.stdin.readline().strip()
        # 문자를 int로 변환 (ord('A')=65, ord('Z')=90)
        encoded = [ord(c) for c in s]
        trie.push(encoded)  # 이제 trie는 Trie[int]로 선언해야 함

    # 4. children_factorial_product 결과 계산 & 출력
    result = trie.children_factorial_product()
    print(result)

if __name__ == "__main__":
    main()