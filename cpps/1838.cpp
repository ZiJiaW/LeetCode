class Solution {
    // 排序以后，滑动窗口
public:
    int maxFrequency(vector<int>& nums, int k) {
        if (nums.size() == 1) return 1;
        sort(begin(nums), end(nums));
        int i = 0, j = 1, res = 1;
        int64_t gap = 0;
        while (j < nums.size()) {
            gap += int64_t(nums[j] - nums[j-1]) * int64_t(j - i);
            if (gap <= int64_t(k)) res = max(res, j - i + 1);
            else {
                while (gap > k) {
                    gap -= nums[j] - nums[i];
                    i++;
                }
            }
            j++;
        }
        return res;
    }
};