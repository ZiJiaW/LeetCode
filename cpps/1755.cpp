/*
分成两半计算可能的组合，排序去重后用滑动窗口逼近goal。
2^(n/2)
子集和问题是NP完全的，因此此题无多项式解法
*/
class Solution {
    vector<int> gen(int s, int e, vector<int>& nums) {
        vector<int> r{0};
        for (int i = s; i < e; i++) {
            int len = r.size();
            for (int j = 0; j < len; j++)
                r.push_back(r[j]+nums[i]);
        }
        sort(begin(r), end(r));
        r.erase(unique(begin(r), end(r)), end(r));
        return r;
    }
public:
    int minAbsDifference(vector<int>& nums, int goal) {
        int n = nums.size();
        if (n == 1) return min(abs(goal), abs(goal-nums[0]));
        auto left = gen(0, n/2, nums), right = gen(n/2, n, nums);
        int res = abs(goal);
        int i = 0, j = right.size() - 1;
        while (i < left.size() && j >= 0) {
            int cur = left[i] + right[j];
            res = min(res, abs(cur - goal));
            if (cur == goal) return 0;
            else if (cur > goal)  j--;
            else i++;
        }
        return res;
    }
};