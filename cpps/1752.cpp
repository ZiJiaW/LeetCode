class Solution {
public:
    bool check(vector<int>& nums) {
        int mid = -1;
        for (int i = 0; i < nums.size()-1; i++) {
            if (nums[i] > nums[i+1]) {
                if (mid != -1) return false;
                else mid = i;
            }
        }
        if (mid == -1) return true;
        else return nums.back() <= nums.front();
    }
};