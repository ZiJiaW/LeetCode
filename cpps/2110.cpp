class Solution
{
public:
    // 简单DP
    long long getDescentPeriods(vector<int> &prices)
    {
        long long res = 1, pre = 1;
        for (int i = 1; i < prices.size(); i++)
        {
            if (prices[i] == prices[i - 1] - 1)
            {
                pre++;
            }
            else
            {
                pre = 1;
            }
            res += pre;
        }
        return res;
    }
};