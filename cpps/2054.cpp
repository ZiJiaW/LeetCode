class Solution
{
    // 本质上就是确定一种遍历顺序，如按照结束时间从小到大遍历的话，就去计算开始时间小于当前活动的最大的value
public:
    int maxTwoEvents(vector<vector<int>> &events)
    {
        sort(begin(events), end(events), [](const auto &a, const auto &b)
             { return a[0] > b[0]; });
        map<int, int> m;
        int cur_max = 0, res = 0;
        for (auto &&e : events)
        {
            cur_max = max(cur_max, e[2]);    // max value over all events whose startTime >= e[0]
            auto it = m.upper_bound(e[1]);   // for current e, find a right event whose startTime > e[1]
            m[e[0]] = max(m[e[0]], cur_max); // means the max value for all e whose startTime >= key
            if (it == m.end())
                res = max(res, e[2]);
            else
                res = max(res, e[2] + it->second);
        }
        return res;
    }

    // 另一种解法，双排序
    // 本质上这种解法更自然一点，分别按照结束时间从大到小排序，和开始时间从大到小排序，然后再遍历1的同时遍历2，记录开始时间大于当前活动的最大value
    int maxTwoEvents2(vector<vector<int>> &events)
    {
        sort(begin(events), end(events), [](const auto &a, const auto &b)
             { return a[1] > b[1]; });
        vector<vector<int>> v(events);
        sort(begin(v), end(v), [](const auto &a, const auto &b)
             { return a[0] > b[0]; });
        int i = 0, maxr = 0, res = 0;
        for (auto &&e : events)
        {
            while (i < v.size() && v[i][0] > e[1])
            {
                maxr = max(maxr, v[i][2]);
                i++;
            }
            res = max(res, e[2] + maxr);
        }
        return res;
    }
};