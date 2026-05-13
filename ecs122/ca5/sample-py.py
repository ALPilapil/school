def compute_min_radius(P, A, k):
    # Your code here. Return the minimum possible d times 60 
    pass

tc = int(input())
results = []
for _ in range(tc):
    n, k = map(int, input().split())
    P = list(map(int, input().split()))
    A = list(map(int, input().split()))
    print(compute_min_radius(P, A, k))