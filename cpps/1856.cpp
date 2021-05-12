class Solution {
    // find the first left one and right one that < than nums[i], and compute min_production
    // use same mono-stac technique in next larger element
public:
    int maxSumMinProduct(vector<int>& nums) {
        vector<int64_t> sums(nums.size(), 0);
        sums[0] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            sums[i] = sums[i-1] + nums[i];
        }
        stack<int> s({0});
        int64_t res = 0;
        int64_t mod = 1e9 + 7;
        vector<int> left(nums.size(), -1);
        vector<int> right(nums.size(), nums.size());
        for (int i = 1; i < nums.size(); i++) {
            while (!s.empty() && nums[s.top()] >= nums[i]) {
                s.pop();
            }
            left[i] = s.empty() ? -1 : s.top();
            s.push(i);
        }
        s = stack<int>();
        s.push(nums.size() - 1);
        for (int i = nums.size() - 2; i >= 0; i--) {
            while (!s.empty() && nums[s.top()] >= nums[i]) {
                s.pop();
            }
            right[i] = s.empty() ? nums.size() : s.top();
            s.push(i);
        }
        for (int i = 0; i < nums.size(); i++) {
            if (left[i] == -1 && right[i] == nums.size()) {
                res = max(res, sums.back() * nums[i]);
            } else if (left[i] == -1) {
                res = max(res, sums[right[i]-1] * nums[i]);
            } else {
                res = max(res, (sums[right[i]-1] - sums[left[i]]) * nums[i]);
            }
        }
        return res % mod;
    }
};