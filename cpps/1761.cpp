/*
三重循环找trio，似乎没有别的好办法了
拿到trio后可以tricky地算degree，三个点的degree减6就行
*/
class Solution {
public:
    int minTrioDegree(int n, vector<vector<int>>& edges) {
        if (n == 2) return -1;
        int e[400][400];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                e[i][j] = 0;
            }
        }
        for (const auto& edge : edges) {
            e[edge[0]-1][edge[1]-1] = 1;
            e[edge[1]-1][edge[0]-1] = 1;
        }
        int deg[400];
        fill(deg, deg+n, 0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                deg[i] += e[i][j];
            }
        }
        int res = INT_MAX;
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                for (int k = j+1; k < n; k++) {
                    if (e[i][j]&&e[i][k]&&e[j][k]) {
                        res = min(res, deg[i]+deg[j]+deg[k]-6);
                    }
                }
            }
        }
        return res == INT_MAX ? -1 : res;
    }
};