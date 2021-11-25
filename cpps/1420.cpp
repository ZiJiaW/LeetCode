class Solution {
    /*
    i: 当前位置
    j: 当前最大值
    r: 当前已用掉的更新次数
    分两种情况dp，在k处刷新最大值 or not，注意边界条件判定~
    */
public:
    int numOfArrays(int n, int m, int k) {
        if (m < k) return 0;
        vector<vector<vector<int>>> dp(n+1, vector<vector<int>>(m+1, vector<int>(k+1, -1)));
        int64_t mod = 1e9 + 7;
        function<int64_t(int,int,int)> dfs = [&](int i, int j, int r) -> int64_t {
            auto &t = dp[i][j][r];
            if (t != -1) return t;
            if (m - j < r || i + r > n) return t = 0;
            if (i == n) return t = 1;
            int64_t cur = static_cast<int64_t>(j) * dfs(i+1, j, r) % mod;
            if (r > 0) {
                for (int x = j+1; x <= m; x++) {
                    cur += dfs(i+1, x, r-1);
                }
            }
            return t = cur % mod;
        };
        return dfs(0, 0, k);
    }
};