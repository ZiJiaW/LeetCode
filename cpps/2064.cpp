class Solution {
    // 常规的贪心+二分查找
public:
    int minimizedMaximum(int n, vector<int>& quantities) {
        int lo = 1, hi = 100000;
        while (lo < hi) {
            int mid = (lo+hi)/2;
            int count = 0;
            for (auto x : quantities) count += x%mid == 0 ? x/mid : x/mid+1;
            if (count > n) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
};