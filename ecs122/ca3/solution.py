import heapq

def bfs(grid, R, C):
    # run A star on the grid

    # always start in (0,0)
    dist = [[-1] * C for _ in range(R)]
    if grid[0][0] == 0:
        return dist
    dist[0][0] = 0
    heap = [(0, 0, 0)]  # (cost, row, col)

    # output should be the minimum amount of time it takes to each cell
    while heap:
        cost, r, c = heapq.heappop(heap)
        if cost > dist[r][c]:
            continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            # however, 0s always turn into -1 and cannot be traversed
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != 0:
                new_cost = cost + grid[nr][nc]
                if dist[nr][nc] == -1 or new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    heapq.heappush(heap, (new_cost, nr, nc))

    return dist

def main():
    # read in file and set I for amount of test cases
    T = int(input())

    # for each test case
    for _ in range(T):

        # read in R, C
        R, C = map(int, input().split())

        # initialize a 2d array of the same R, C dimensions
        grid = [list(map(int, input().split())) for _ in range(R)]

        # calculate the minimum time to move to each cell via bfs
        result = bfs(grid, R, C)
        for row in result:
            print(*row)

    return 0

if __name__ == "__main__":
    main()
