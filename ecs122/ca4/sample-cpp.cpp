#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v, w;
    Edge(){}
    Edge(int a, int b, int c) { u = a; v = b; w = c;}
};


/* Complete this function:
    Given the weights of vertices and edges of the graph, compute the minimum cost
    weights[i] = weight of vertex i (ignore weights[0])
    The i^{th} edge (0-indexed) is between (edges[i].u, edges[i].v) and has weight edges[i].w
    edges is a vector of size m, (i.e., the edges are described by edges[0], edges[1], ... edges[m-1])
*/
long long computeMinimumCost(vector<int> weights, vector<Edge> edges) {
    int n = (int)weights.size() - 1; // weights[i] = the weight of vertex i, please ignore weights[0].
    long long cost = 0; // use 64-bit integer for the answer
    return cost;
}

/* The following lines take care of input and output for you. You may ignore this section.
*/
int main() {
    ios_base::sync_with_stdio(0); cin.tie(0);
    int tc;
    cin >> tc;
    for (int t = 0; t < tc; t++) {
        int n, m;
        cin >> n >> m;
        vector<int> weights(n + 1);
        vector<Edge> edges;
        for (int i=1;i<=n;++i) cin >> weights[i];
        for (int i=0;i<m;++i) {
            int u, v, w;
            cin >> u >> v >> w;
            edges.push_back(Edge(u, v, w));
        }
        cout << computeMinimumCost(weights, edges) << '\n';
    }
}
