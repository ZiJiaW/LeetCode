class Solution
{
    // 暴力尝试所有组合，通过预处理减少check validation的消耗
public:
    int maximumGood(vector<vector<int>> &statements)
    {
        int res = 0;
        const int n = statements.size();
        vector<vector<int>> good(n, vector<int>{});
        vector<vector<int>> bad(n, vector<int>{});
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (statements[i][j] == 0)
                    bad[i].push_back(j);
                if (statements[i][j] == 1)
                    good[i].push_back(j);
            }
        }
        for (int mask = 1; mask < 1 << n; mask++)
        {
            for (int i = 0; i < n; i++)
            {
                for (auto j : good[i])
                {
                    if (mask & (1 << i))
                    {
                        if (!(mask & (1 << j)))
                        {
                            goto end;
                        }
                    }
                }
                for (auto j : bad[i])
                {
                    if (mask & (1 << i))
                    {
                        if (mask & (1 << j))
                        {
                            goto end;
                        }
                    }
                }
            }
            res = max(res, __builtin_popcount(mask));
        end:;
        }
        return res;
    }
};