import sys


def bisect_right(A, x, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if A[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def can_cover(A, P, n, k, q):
    uncovered = 0   # index of the leftmost city not yet covered
    towers_used = 0 # how many towers we've placed so far

    # keep going until every city is covered
    while uncovered < n:
        # if we've already used all k towers and still have uncovered cities, fail
        # setting provided were insufficient 
        if towers_used == k:
            return False
        towers_used += 1  # place one more tower


        # establish the right most tower location 
        # identify the range possible for the sttings provided  
        max_right_pos = (60 * A[uncovered] + q) // 60
        # j_max: last index in A that is <= max_right_pos
        j_max = bisect_right(A, max_right_pos, uncovered, n) - 1


        # best reach set to -1 initially
        # nothing has been reached yet
        best_reach = -1
        for j in range(uncovered, j_max + 1):
            # 6000 to scale everything
            # seeing if tower being placed at j will have it reach the leftmost uncovered city
            if 6000 * (A[j] - A[uncovered]) <= q * P[j]:
                reach = 6000 * A[j] + q * P[j]
                if reach > best_reach:
                    best_reach = reach  # keep track of the farthest-reaching valid tower

        # keep track of how much has been covered
        covered_up_to = best_reach // 6000
        uncovered = bisect_right(A, covered_up_to, uncovered, n)

    return True  # all cities covered within k towers


def compute_min_radius(A, P, k):
    n = len(A)  # number of cities

    # return radius 0 if there are more towers than cities
    if k >= n:
        return 0

    # binary search on q = 60*d (the scaled radius the problem asks us to output)
    lo = 0  # minimum possible q
    # upper bound: worst case is 1 tower with minimum power p=20 must cover the full span resulting in a large d
    # this implies that span <= d * 20/100 = d/5, so d >= 5*span, so q = 60*d >= 300*span, remember 1 is just the answer adjustment
    # q_max = 300 * span
    hi = 300 * (A[n - 1] - A[0]) + 1  # +1 makes hi an exclusive upper bound
    
    # binary search
    while lo < hi:
        mid = (lo + hi) // 2         # candidate q value to test
        if can_cover(A, P, n, k, mid):
            hi = mid                 # mid works, try a smaller radius
        else:
            lo = mid + 1             # mid too small, need a larger radius

    # lo is the smallest q = 60*d that allows full coverage with k towers
    return lo


def main():
    # read all input at once and split into tokens for fast parsing
    data = sys.stdin.read().split()
    idx = 0  # current position in the token list

    tc = int(data[idx]); idx += 1   # number of test cases
    results = []

    for _ in range(tc):
        # read n (number of cities) and k (number of towers allowed)
        n, k = int(data[idx]), int(data[idx + 1]); idx += 2
        # read power percentages for each city (values are in {20, 40, 60, 80, 100})
        P = [int(data[idx + i]) for i in range(n)]; idx += n
        # read city positions in increasing order (guaranteed sorted)
        A = [int(data[idx + i]) for i in range(n)]; idx += n
        results.append(compute_min_radius(A, P, k))

    # print all answers, one per line
    print('\n'.join(map(str, results)))


if __name__ == "__main__":
    main()
