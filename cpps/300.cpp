class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        // 这题的nlgn解法一点都不medium
        // 详情见https://www.geeksforgeeks.org/longest-monotonically-increasing-subsequence-size-n-log-n/
        // 用数组r[i]表示largest end element among all monotonically increasing subsequence of length i+1
        int res = 1, idx = 1;
        vector<int> r(nums.size(), 0);
        r[0] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] < r[0]) {
                r[0] = nums[i];
            } else if (nums[i] > r[idx-1]) {
                r[idx] = nums[i];
                idx++;
            } else {
                int lo = 0, hi = idx-1;
                while (lo != hi) {
                    int mid = lo + (hi-lo)/2;
                    if (r[mid] >= nums[i]) hi = mid;
                    else lo = mid+1;
                }
                r[lo] = nums[i];
            }
        }
        return idx;
    }
};