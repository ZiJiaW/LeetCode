class Solution
{
    // 记录需要填多少个0，使得构成连续的1
    // 实际上只需要计数1的数量len，然后按len遍历，记录其中出现的0的最小数量
public:
    int minSwaps(vector<int> &nums)
    {
        int len = nums.size();
        int count = 0;
        for (auto n : nums)
            count += n == 1;
        if (count == 0)
            return 0;
        nums.reserve(2 * len);
        for (int i = 0; i < len; i++)
        {
            nums.push_back(nums[i]);
        }
        int l = 0, r = 0, t = 0;
        int res = len;
        while (r < nums.size())
        {
            t += nums[r] == 0;
            if (r - l + 1 == count)
            {
                res = min(res, t);
                if (nums[l] == 0)
                {
                    t--;
                }
                l++;
            }
            r++;
        }
        return res;
    }
};