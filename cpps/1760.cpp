class Solution {
public:
    int minimumSize(vector<int>& nums, int maxOperations) {
        int64_t sum = 0, lo = 0, hi = 0;
        for (int64_t num : nums) { 
            sum += num;
            hi = max(hi, num);
        }
        lo = sum / (maxOperations + nums.size()) - 1;
        if (lo < 0) lo = 0;
        while (lo + 1 < hi) {
            int mid = lo + (hi-lo)/2;
            int t = 0;
            for (auto num : nums) {
                if (num <= mid) continue;
                t += num%mid == 0 ? num/mid-1 : num/mid;
            }
            if (t > maxOperations) lo = mid;
            else hi = mid;
        }
        return hi;
    }
};