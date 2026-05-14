import sys
from bisect import bisect_right  # efficient binary search on sorted lists

# The problem asks us to output q = 60*d instead of d directly,
# because d may be fractional (e.g. d = 100/3) but 60*d is always an integer.
# So everywhere below, q = 60*d is the quantity we work with.
#
# The raw coverage condition from the problem is:
#   |A[j] - A[c]| <= d * P[j] / 100
# Substituting d = q/60 and multiplying both sides by 6000 (= 60 * 100):
#   6000 * |A[j] - A[c]| <= q * P[j]
# That's where the magic number 6000 comes from — it clears both the /60 and the /100.


# check if k towers can cover all cities given q = 60*d
# greedy strategy: scan left to right, and whenever we find an uncovered city,
# place the tower that covers it and extends coverage as far right as possible
def can_cover(A, P, n, k, q):
    uncovered = 0   # index of the leftmost city not yet covered
    towers_used = 0 # how many towers we've placed so far

    # keep going until every city is covered
    while uncovered < n:
        # if we've already used all k towers and still have uncovered cities, fail
        if towers_used == k:
            return False
        towers_used += 1  # place one more tower

        # find the rightmost index j_max whose tower could possibly cover city[uncovered]
        # coverage requires A[j] - A[uncovered] <= d * P[j]/100
        # worst case is P[j]=100 (max power), giving A[j] - A[uncovered] <= d = q/60
        # so A[j] <= A[uncovered] + q/60, i.e. 60*A[j] <= 60*A[uncovered] + q
        # the 60 here cancels the /60 from d = q/60
        max_right_pos = (60 * A[uncovered] + q) // 60
        # j_max: last index in A that is <= max_right_pos
        j_max = bisect_right(A, max_right_pos, uncovered, n) - 1

        # among all candidate towers [uncovered .. j_max], find the one that:
        #   1. actually covers city[uncovered] (coverage check using 6000 scaling)
        #   2. has the maximum right reach (greedy: push coverage as far right as possible)
        best_reach = -1
        for j in range(uncovered, j_max + 1):
            # scaled coverage check: 6000 * (A[j] - A[uncovered]) <= q * P[j]
            # derived from |A[j] - A[c]| <= d * P[j]/100, multiplied through by 6000
            if 6000 * (A[j] - A[uncovered]) <= q * P[j]:
                # right reach of tower j, scaled by 6000 to stay integer:
                # tower covers up to A[j] + d*P[j]/100 = A[j] + q*P[j]/6000
                # multiply by 6000: 6000*A[j] + q*P[j]
                reach = 6000 * A[j] + q * P[j]
                if reach > best_reach:
                    best_reach = reach  # keep track of the farthest-reaching valid tower

        # convert the scaled reach back to position space:
        # city c is covered iff 6000*A[c] <= best_reach, i.e. A[c] <= best_reach // 6000
        covered_up_to = best_reach // 6000
        # jump uncovered to the first city whose position exceeds covered_up_to
        uncovered = bisect_right(A, covered_up_to, uncovered, n)

    return True  # all cities covered within k towers


def compute_min_radius(A, P, k):
    n = len(A)  # number of cities

    # if we have at least as many towers as cities, place one per city — radius 0 works
    if k >= n:
        return 0

    # binary search on q = 60*d (the scaled radius the problem asks us to output)
    lo = 0  # minimum possible q
    # upper bound: worst case is 1 tower with minimum power p=20 must cover the full span
    # coverage: span <= d * 20/100 = d/5, so d >= 5*span, so q = 60*d >= 300*span
    # 300 = 60 * (100/20), where 60 scales d and 100/20 inverts the min power fraction
    hi = 300 * (A[n - 1] - A[0]) + 1  # +1 makes hi an exclusive upper bound

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
