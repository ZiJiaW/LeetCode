class Solution
{
    // case1 计算长度大于2的环 其最大长度为maxcycle
    // case2 计算长度为2的环以及链接到其上的最长的两条边，这一步可以把图中的有向边取逆来计算
    // case2中的单元可共存（画图验证）
    // 返回两种case的最大值
public:
    int maximumInvitations(vector<int> &f)
    {
        // compute max cycle
        vector<int> v(f.size(), 0);
        int maxcycle = 0;
        function<void(int, int)> dfs = [&](int i, int p)
        {
            v[i] = p;
            if (v[f[i]] == 0)
                dfs(f[i], p + 1);
            else if (v[f[i]] > 0)
                maxcycle = max(maxcycle, p - v[f[i]] + 1);
            v[i] = -p;
        };
        for (int i = 0; i < f.size(); i++)
        {
            if (v[i] == 0)
                dfs(i, 1);
        }
        // compute 2 cycles
        vector<vector<int>> adj(f.size(), vector<int>{});
        int maxcyclelen2 = 0;
        for (int i = 0; i < f.size(); i++)
            adj[f[i]].push_back(i);
        function<int(int, int)> getMaxLen = [&](int i, int omit)
        {
            int res = 0;
            for (auto j : adj[i])
            {
                if (j != omit)
                    res = max(res, getMaxLen(j, omit) + 1);
            }
            return res;
        };
        for (int i = 0; i < f.size(); i++)
        {
            if (i == f[f[i]] && v[i] != 0)
            {
                maxcyclelen2 += 2 + getMaxLen(i, f[i]) + getMaxLen(f[i], i);
                v[f[i]] = 0;
            }
        }
        return max(maxcyclelen2, maxcycle);
    }
};