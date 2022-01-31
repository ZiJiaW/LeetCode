class Solution
{
    // 计算n台电脑能否运行mid时间
    // 若一堆电池的长度均不超过平均数，则其和sum/n为最长运行时间
public:
    long long maxRunTime(int n, vector<int> &batteries)
    {
        long long lo = 0;
        long long hi = accumulate(begin(batteries), end(batteries), (long long)0) / n + 1;
        while (lo < hi - 1)
        {
            long long mid = lo + (hi - lo) / 2;
            long long t = 0;
            for (auto i : batteries)
                t += i <= mid ? i : mid;
            if (t < n * mid)
                hi = mid;
            else
                lo = mid;
        }
        return lo;
    }
};