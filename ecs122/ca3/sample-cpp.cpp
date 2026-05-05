#include <iostream>
#include <vector>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int T;
    cin >> T;
    while (T--) {
        int R, C;
        cin >> R >> C;
        vector<vector<int>> grid(R, vector<int>(C));
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                cin >> grid[i][j];
            }
        }
        vector<vector<int>> result(R, vector<int>(C, -1));
        /*
            Your code here to compute result[i][j] as the minimum amount of time ADA needs to reach (i + 1, j + 1) from (1, 1)
        */
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                cout << result[i][j] << (j == C - 1 ? '\n' : ' ');
            }
        }
    }
}