class Solution
{
public:
    int findFinalValue(vector<int> &nums, int original)
    {
        vector<bool> v(1001, false);
        for (auto n : nums)
            v[n] = true;
        while (original <= 1000 && v[original])
        {
            original *= 2;
        }
        return original;
    }
};