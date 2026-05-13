#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

long long compute_min_radius(vector<int> &P, vector<long long> &A, int k) {
    // Your code here. Return the minimum possible d times 60
}


int main() {
    ios_base::sync_with_stdio(0); cin.tie(0);
    int tc;
    cin >> tc;
    for (int t = 0; t < tc; t++) {
        int n, k;
        cin >> n >> k;
        vector<int> P(n);
        vector<long long> A(n);
        for (int i = 0; i < n; i++) {
            cin >> P[i];
        }
        for (int i = 0; i < n; i++) {
            cin >> A[i];
        }
        cout << compute_min_radius(P, A, k) << '\n';
    }
}
