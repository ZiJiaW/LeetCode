class Solution {
    // dp 记录以当前位置为顶的最大金字塔高度
    // 那么它的金字塔高度等于其左下和右下为顶的金字塔高度加上1（当然必须保证当前位置以及下面的位置为1）
public:
    int countPyramids(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), res = 0;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp.back() = grid.back();
        auto count = [&]() {
            int res = 0;
            for (int i = m-2; i >= 0; i--) {
                for (int j = 0; j < n; j++) {
                    if (grid[i][j] == 1 && grid[i+1][j] == 1 && j-1 >= 0 && j+1 < n 
                        && dp[i+1][j-1] >= 1 && dp[i+1][j+1] >= 1) {
                        dp[i][j] = min(dp[i+1][j-1], dp[i+1][j+1]) + 1;
                    } else {
                        dp[i][j] = grid[i][j];
                    }
                    res += max(0, dp[i][j] - 1);
                }
            }
            return res;
        };
        int r = count();
        reverse(begin(grid), end(grid));
        dp.back() = grid.back();
        return r + count();
    }
};