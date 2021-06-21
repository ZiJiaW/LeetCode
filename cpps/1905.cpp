class Solution {
public:
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
        auto m = grid1.size(), n = grid1[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid2[i][j] == 1 && grid1[i][j] == 0)  grid1[i][j] = 2;
                else grid1[i][j] = grid1[i][j] & grid2[i][j];
            }
        }
        vector<vector<bool>> v(m, vector<bool>(n, false));
        vector<vector<int>> d{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        bool isValid = true;
        function<void(int,int)> dfs = [&](int i, int j) {
            if (v[i][j]) return;
            v[i][j] = true;
            for (auto &dr:d) {
                auto ni = i + dr[0], nj = j + dr[1];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid1[ni][nj] != 0) {
                    if (grid1[ni][nj] == 2) {
                        isValid = false;
                        continue;
                    }
                    dfs(ni, nj);
                }
            }
        };
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!v[i][j] && grid1[i][j] == 1) {
                    isValid = true;
                    dfs(i, j);
                    if (isValid) res++;
                }
            }
        }
        return res;
    }
};