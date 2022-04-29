class Solution
{
public:
    vector<int> maxScoreIndices(vector<int> &nums)
    {
        vector<int> res;
        int n = nums.size(), cur = 0;
        vector<int> sum0(n, 0), sum1(n, 0);
        for (int i = 0; i < n; i++)
        {
            if (i == 0)
            {
                sum0[i] = nums[i] == 0;
                sum1[i] = nums[i] == 1;
            }
            else
            {
                sum0[i] = sum0[i - 1] + (nums[i] == 0);
                sum1[i] = sum1[i - 1] + (nums[i] == 1);
            }
        }
        for (int i = 0; i <= n; i++)
        {
            int left = i == 0 ? 0 : sum0[i - 1];
            int right = i == n ? 0 : i == 0 ? sum1[n - 1]
                                            : sum1[n - 1] - sum1[i - 1];
            if (left + right > cur)
            {
                cur = left + right;
                res.clear();
                res.push_back(i);
            }
            else if (left + right == cur)
            {
                res.push_back(i);
            }
        }
        return res;
    }
};