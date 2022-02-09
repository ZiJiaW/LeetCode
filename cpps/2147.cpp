class Solution
{
    // 更简单的解法是计算相邻座椅之间可以放置的隔板数量乘积
public:
    int numberOfWays(string corridor)
    {
        int64_t mod = 1e9 + 7, num_S = 0;
        for (auto c : corridor)
            num_S += c == 'S';
        if (num_S % 2 == 1 || num_S == 0)
            return 0;
        while (corridor.back() == 'P')
            corridor.pop_back();
        vector<vector<int64_t>> dp(3, vector<int64_t>(corridor.size(), -1));
        function<int64_t(int, int)> dfs = [&](int i, int left)
        {
            if (i == corridor.size())
            {
                return left == 0 ? 1l : 0l;
            }
            if (dp[left][i] >= 0)
                return dp[left][i];
            int64_t res = 0;
            if (left == 0)
            {
                if (corridor[i] == 'S')
                {
                    res = dfs(i + 1, 1);
                }
                else
                {
                    res = dfs(i + 1, 0);
                }
            }
            else if (left == 1)
            {
                if (corridor[i] == 'S')
                {
                    res = dfs(i + 1, 0) + dfs(i + 1, 2);
                }
                else
                {
                    res = dfs(i + 1, 1);
                }
            }
            else
            {
                if (corridor[i] == 'S')
                {
                    res = 0;
                }
                else
                {
                    res = dfs(i + 1, 0) + dfs(i + 1, 2);
                }
            }
            return dp[left][i] = res % mod;
        };
        return dfs(0, 0);
    }
};