class Solution {
public:
    vector<int> minDifference(vector<int>& nums, vector<vector<int>>& queries) {
        int n = nums.size();
        vector<vector<int>> c(101, vector<int>(n, 0));
        for (int i = 0; i < n; i++) {
            for (int k = 1; k <= 100; k++) {
                c[k][i] = int(nums[i] == k);
                if (i > 0) c[k][i] += c[k][i-1];
            }
        }
        vector<int> res;
        res.reserve(queries.size());
        for (auto& q : queries) {
            int pre = -1, r = INT_MAX;
            for (int k = 1; k <= 100; k++) {
                int count = q[0] > 0 ? c[k][q[1]] - c[k][q[0]-1] : c[k][q[1]];
                if (count == 0) continue;
                if (pre == -1) pre = k;
                else {
                    r = min(r, k - pre);
                    pre = k;
                }
            }
            if (r == INT_MAX) r = -1;
            res.push_back(r);
        }
        return res;
    }
};