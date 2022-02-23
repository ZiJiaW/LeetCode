class Solution
{
public:
    vector<int> pivotArray(vector<int> &nums, int pivot)
    {
        vector<int> res(nums.size(), -1);
        int l = 0, r = nums.size() - 1;
        for (int i = 0; i < nums.size(); i++)
        {
            if (nums[i] < pivot)
                res[l++] = nums[i];
            if (nums[nums.size() - 1 - i] > pivot)
                res[r--] = nums[nums.size() - 1 - i];
        }
        while (l <= r)
            res[l++] = pivot;
        return res;
    }
};