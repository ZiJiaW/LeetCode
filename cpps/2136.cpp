class Solution
{
    // 最优解：优先准备growTime最长的花
public:
    int earliestFullBloom(vector<int> &plantTime, vector<int> &growTime)
    {
        int n = plantTime.size();
        vector<pair<int, int>> v;
        v.reserve(n);
        for (int i = 0; i < n; i++)
        {
            v.emplace_back(growTime[i], plantTime[i]);
        }
        sort(rbegin(v), rend(v));
        int res = 0, p = 0;
        for (auto &i : v)
        {
            p += i.second;
            res = max(res, p + i.first);
        }
        return res;
    }
};