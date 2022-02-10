class Solution
{
public:
    vector<int> rearrangeArray(vector<int> &nums)
    {
        vector<int> res(nums.size(), 0);
        int i = 0, j = 1;
        for (auto n : nums)
        {
            if (n > 0)
            {
                res[i] = n;
                i += 2;
            }
            else
            {
                res[j] = n;
                j += 2;
            }
        }
        return res;
    }
};