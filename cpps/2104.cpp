class Solution
{
public:
    long long subArrayRanges(vector<int> &nums)
    {
        long long res = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            int s = INT_MAX, b = INT_MIN;
            for (int j = i; j < nums.size(); j++)
            {
                s = min(s, nums[j]);
                b = max(b, nums[j]);
                res += b - s;
            }
        }
        return res;
    }
};