class Solution {
public:
    int minimumDeletions(vector<int>& nums) {
        int min_elem = INT_MAX, min_idx = -1;
        int max_elem = INT_MIN, max_idx = -1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] > max_elem) {
                max_elem = nums[i];
                max_idx = i;
            }
            if (nums[i] < min_elem) {
                min_elem = nums[i];
                min_idx = i;
            }
        }
        int l = min(min_idx, max_idx), r = max(min_idx, max_idx);
        vector<int> v{r+1, static_cast<int>(nums.size()-l), static_cast<int>(l+1+nums.size()-r)};
        return *min_element(begin(v), end(v));
    }
};