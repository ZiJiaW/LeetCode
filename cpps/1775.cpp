class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        if (m*6 < n || n*6 < m) return -1;
        int sum1 = accumulate(begin(nums1), end(nums1), 0);
        int sum2 = accumulate(begin(nums2), end(nums2), 0);
        if (sum1 > sum2) {
            swap(nums1, nums2);
        }
        int diff = abs(sum2 - sum1), res = 0;
        vector<int> count(6, 0);
        // 计数可减少diff的操作数量
        for (auto x : nums1) count[6-x]++;// 对小的计数增加至6
        for (auto x : nums2) count[x-1]++;// 对大的计数减少至1
        for (int i = 5; i > 0 && diff > 0; i--) {
            int opCount = min(count[i], diff/i + int(diff%i!=0));
            res += opCount;
            diff -= opCount * i;
        }
        return res;
    }
};