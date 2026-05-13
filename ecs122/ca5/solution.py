import sys
from bisect import bisect_right

input = sys.stdin.readline


# given a fixed q (= 60*d), check if k towers can cover all cities
# strategy: greedily scan left to right
#   - when we hit an uncovered city, find the tower location that
#     can still cover it (from the right) while reaching as far right as possible
#   - the coverage condition for tower j covering city i:
#       6000 * (A[j] - A[i]) <= q * p[j]
#   - the right reach of tower j (scaled by 6000 to stay integer):
#       reach = 6000*A[j] + q*p[j]
#   - city c is covered if 6000*A[c] <= reach
def can_cover(A, P, n, k, q):
    uncovered = 0
    towers_used = 0

    while uncovered < n:
        # need one more tower to cover city at index `uncovered`
        if towers_used == k:
            return False
        towers_used += 1

        # upper bound on j: even with p=100, tower can only reach leftward by q/60
        # so A[j] - A[uncovered] <= q/60, meaning 60*A[j] <= 60*A[uncovered] + q
        max_right_pos = (60 * A[uncovered] + q) // 60
        j_max = bisect_right(A, max_right_pos, uncovered, n) - 1

        # find the tower in [uncovered, j_max] that covers city[uncovered]
        # and has the maximum right reach
        best_reach = -1
        for j in range(uncovered, j_max + 1):
            # check: can tower at j cover city[uncovered] going leftward?
            if 6000 * (A[j] - A[uncovered]) <= q * P[j]:
                reach = 6000 * A[j] + q * P[j]
                if reach > best_reach:
                    best_reach = reach

        # find the first city not covered by the best tower
        # city c is covered iff 6000*A[c] <= best_reach
        # since A[c] is integer: A[c] <= best_reach // 6000
        covered_up_to = best_reach // 6000
        uncovered = bisect_right(A, covered_up_to, uncovered, n)

    return True


def compute_min_radius(P, A, k):
    n = len(A)

    # edge case: k >= n means we can put a tower in every city, radius 0 works
    if k >= n:
        return 0

    # binary search on q = 60*d
    # lower bound: 0 (might work if k >= n, already handled)
    # upper bound: worst case is 1 tower with p=20 covering entire span
    #   need d * 0.2 >= span, so d <= 5*span, q <= 300*span
    lo = 0
    hi = 300 * (A[n - 1] - A[0]) + 1

    while lo < hi:
        mid = (lo + hi) // 2
        if can_cover(A, P, n, k, mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


tc = int(input())
out = []
for _ in range(tc):
    n, k = map(int, input().split())
    P = list(map(int, input().split()))
    A = list(map(int, input().split()))
    out.append(compute_min_radius(P, A, k))

print('\n'.join(map(str, out)))
