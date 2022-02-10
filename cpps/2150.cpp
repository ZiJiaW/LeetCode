class Solution
{
public:
    vector<int> findLonely(vector<int> &nums)
    {
        if (nums.size() == 1)
            return nums;
        vector<int> res;
        sort(begin(nums), end(nums));
        for (int i = 1; i < nums.size() - 1; i++)
        {
            if (nums[i] > nums[i - 1] + 1 && nums[i] < nums[i + 1] - 1)
                res.push_back(nums[i]);
        }
        if (nums[0] < nums[1] - 1)
        {
            res.push_back(nums.front());
        }
        if (nums[nums.size() - 1] > nums[nums.size() - 2] + 1)
        {
            res.push_back(nums.back());
        }
        return res;
    }
};