class Solution
{ // 暴力尝试所有的k，注意先筛选一下k，用最小元素和最大元素
public:
    vector<int> recoverArray(vector<int> &nums)
    {
        sort(begin(nums), end(nums));
        unordered_map<int, int> count;
        for (auto x : nums)
            count[x]++;
        unordered_set<int> s;
        for (int i = 0; i < nums.size(); i++)
        {
            for (int j = i + 1; j < nums.size(); j++)
            {
                int diff = nums[j] - nums[i];
                if (diff & 0x1)
                    continue;
                if (diff == 0)
                    continue;
                if (count.find(nums[0] + diff) == count.end() || count.find(nums.back() - diff) == count.end())
                    continue;
                s.insert(diff);
            }
        }
        vector<int> res;
        for (int k : s)
        {
            unordered_map<int, int> c = count;
            for (auto n : nums)
            {
                if (c[n] > 0)
                {
                    if (c.find(n + k) == c.end() || c[n + k] == 0)
                    {
                        goto fail;
                    }
                    c[n]--;
                    c[n + k]--;
                }
            }
            c = count;
            for (auto n : nums)
            {
                if (c[n] > 0)
                {
                    res.push_back(n + k / 2);
                    c[n]--;
                    c[n + k]--;
                }
            }
            return res;
        fail:;
        }
        return res;
    }
};