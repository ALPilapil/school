T = int(input())
for _ in range(T):
    R, C = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(R)]
    result = [ [-1 for _ in range(C)] for _ in range(R)]

    # Your code here to compute result[i][j] as the minimum amount of time ADA needs to reach (i + 1, j + 1) from (1, 1)

    for row in result:
        print(*row, sep=' ')

