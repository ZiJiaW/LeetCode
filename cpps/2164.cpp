class Solution
{
public:
    vector<int> sortEvenOdd(vector<int> &nums)
    {
        vector<int> odds, evens;
        for (int i = 0; i < nums.size(); i++)
        {
            if (i & 1)
                odds.push_back(nums[i]);
            else
                evens.push_back(nums[i]);
        }
        sort(rbegin(odds), rend(odds));
        sort(begin(evens), end(evens));
        for (int i = 0; i < nums.size(); i++)
        {
            if (i & 1)
                nums[i] = odds[i / 2];
            else
                nums[i] = evens[i / 2];
        }
        return nums;
    }
};