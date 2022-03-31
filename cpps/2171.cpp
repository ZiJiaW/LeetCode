class Solution
{
public:
    long long minimumRemoval(vector<int> &beans)
    {
        int64_t sum = 0;
        priority_queue<int, vector<int>, greater<int>> q;
        int64_t res = numeric_limits<int64_t>::max(), removed = 0;
        for (auto n : beans)
        {
            sum += n;
            q.push(n);
        }
        while (!q.empty())
        {
            int64_t t = q.top();
            q.pop();
            sum -= t;
            res = min(res, sum - t * int64_t(q.size()) + removed);
            removed += t;
        }
        return res;
    }
};