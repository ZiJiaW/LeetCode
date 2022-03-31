class Solution
{
    // simple dp
public:
    int maximumANDSum(vector<int> &nums, int numSlots)
    {
        int res = 0;
        unordered_map<int, int> dp;
        function<int(int, int)> dfs = [&](int idx, int mask)
        {
            int key = (idx << 24) | mask;
            if (dp.find(key) != dp.end())
                return dp[key];
            if (idx == nums.size())
                return 0;
            int tmp = 0;
            for (int j = 0; j < numSlots; j++)
            {
                if (mask & (1 << (2 * j + 1)))
                    continue;
                if (mask & (1 << (2 * j)))
                {
                    tmp = max(tmp, (nums[idx] & (j + 1)) + dfs(idx + 1, mask | (1 << (2 * j + 1))));
                }
                else
                {
                    tmp = max(tmp, (nums[idx] & (j + 1)) + dfs(idx + 1, mask | (1 << (2 * j))));
                }
            }
            dp[key] = tmp;
            return tmp;
        };
        return dfs(0, 0);
    }
};