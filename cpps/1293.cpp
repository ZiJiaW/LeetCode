class Solution {
    // 广搜
public:
    int shortestPath(vector<vector<int>>& grid, int k) {
        int m = grid.size(), n = grid[0].size();
        queue<tuple<int,int,int>> q({make_tuple(0, 0, k)});
        vector<vector<vector<bool>>> v(m, vector<vector<bool>>(n, vector<bool>(m*n, false)));
        vector<vector<int>> d{{1,0},{0,1},{-1,0},{0,-1}};
        int step = 0;
        v[0][0][k-1] = true;
        while (!q.empty()) {
            int len = q.size();
            while (len-- > 0) {
                auto [x, y, r] = q.front(); q.pop();
                if (x == m-1 && y == n-1) return step;
                for (auto &dd : d) {
                    int nx = x+dd[0], ny = y+dd[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                        if (grid[nx][ny] == 0) {
                            if (!v[nx][ny][r]) {
                                q.push({nx, ny, r});
                                v[nx][ny][r] = true;
                            }
                        } else if (r > 0) {
                            if (!v[nx][ny][r-1]) {
                                q.push({nx, ny, r-1});
                                v[nx][ny][r-1] = true;
                            }
                        }
                    }
                }
            }
            step++;
        }
        return -1;
    }
};