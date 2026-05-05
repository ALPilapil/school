''' Complete the following function:
    Given the weights of vertices and edges of the graph, compute the minimum cost
    weights[i] = weight of vertex i (ignore weights[0])
    The i^{th} edge (0-indexed) is between (edges[i].u, edges[i].v) and has weight edges[i].w
    edges is a vector of size m, (i.e., the edges are described by edges[0], edges[1], ... edges[m-1])
'''
def computeMinimumCost(weights, edges):
    n = len(weights) - 1  # ignore weights[0], weights[1], weights[2], ... weights[n] denote the weights of the vertices
    # The i^{th} edge (0-indexed) has ends edges[i][0], edges[i][1] and weight edges[i][2]
    # edges is a list of tuples of size three
    m = len(edges)
    cost = 0
    return cost

# The following lines take care of input and output for you. You may ignore this section.

T = int(input())
for _ in range(T):
    n, m = map(int, input().split())
    weights = list(map(int, ("0 " + input()).split())) # Adding 0 to index weights by 1, instead of 0
    edges = []
    for _ in range(m):
        u, v, w  = map(int, input().split(' '))
        edges.append((u, v, w))
    print(computeMinimumCost(weights, edges))
