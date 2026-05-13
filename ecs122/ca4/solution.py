from heapq import heapify, heappush, heappop
import sys
input = sys.stdin.readline

def buildByteLandWaterSystem(n, wells, roads):
    # wells is a list, each element is either cost of well if + or no well if -1
    # roads is a list of 3 tuples: (village_u, village_v, rebuild_cost)
    # - one indexed
    # wells[i] corresponds to village[i+1], tells you if there is a well there
    
    # initialize n+1 empty lists as the list of adjacency lists
    adj_list = [[] for i in range(n+1)]

    # loop through roads
    for road in roads:
        u, v, cost = road
        # bidirectional so have to add costs both ways for v and u
        adj_list[u].append((cost, v))
        adj_list[v].append((cost, u))
    
    # loop through wells
    for i in range(len(wells)):
        if wells[i] == -1:
            # no well, skip
            continue
        # connect the root to all the wells
        adj_list[0].append((wells[i], i+1))
        # connect all the wells to the root
        adj_list[i+1].append((wells[i], 0))

    # algorithm
    # intialize empty heap
    heap = [(0,0)]
    visited = set()
    total_cost = 0

    while heap:
        # pop the lowest cost node
        cost, node = heappop(heap)
        if node in visited:
            # if it's already visited skip
            continue
        # otherwise, add it to visited nodes
        visited.add(node)
        # increae the cost accordingly
        total_cost += cost
        # check the neighbors
        for pair in adj_list[node]:
            edge_cost, neighbor = pair
            if neighbor not in visited:
                # push the non visited neighbors to the heap
                # min heap by default, so next iteration will be checking the lowest cost
                heappush(heap, (edge_cost, neighbor))

    return total_cost

def main():
    # getting input
    C = int(input())
    for _ in range(C):
        n, m = map(int, input().split())
        wells = list(map(int, input().split()))  # wells[i] is cost for village i+1, or -1
        roads = []
        for _ in range(m):
            u, v, c = map(int, input().split())
            roads.append((u, v, c))
        print(buildByteLandWaterSystem(n, wells, roads))

if __name__ == "__main__":
    main()
    # used this to help: https://walkccc.me/LeetCode/problems/1168/#__tabbed_1_3
    # - i was insipired to use Prim's algorithm as they did here
    # - i implemented it very similarly with a heapq, required for Prim's
    # - also used their idea of using a root node to connect things
