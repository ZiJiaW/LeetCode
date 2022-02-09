class Solution
{
    // 计算序列中的最大和最小可能值
public:
    int numberOfArrays(vector<int> &differences, int lower, int upper)
    {
        int res = 0;
        int64_t acc = 0, max_diff = 0, min_diff = 0;
        for (auto x : differences)
        {
            acc += x;
            max_diff = max(acc, max_diff);
            min_diff = min(acc, min_diff);
        }
        for (int i = lower; i <= upper; i++)
        {
            res += i + max_diff <= upper && i + min_diff >= lower;
        }
        return res;
    }
};