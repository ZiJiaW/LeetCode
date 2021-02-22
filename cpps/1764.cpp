class Solution {
    int match(vector<int>& nums, int i, vector<int>& group) {
        for (int k = i; k < nums.size(); k++) {
            if (nums.size() - k < group.size()) return -1;
            bool ok = true;
            for (int j = 0; j < group.size(); j++) {
                if (nums[k+j] != group[j]) {
                    ok = false;
                    break;
                }
            }
            if (ok) return k;
        }
        return -1;
    }
public:
    bool canChoose(vector<vector<int>>& groups, vector<int>& nums) {
        int i = 0;
        for (auto& group : groups) {
            int r = match(nums, i, group);
            if (r == -1) return false;
            i = r + group.size();
        }
        return true;
    }
};