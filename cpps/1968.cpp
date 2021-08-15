class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        sort(begin(nums), end(nums));
        vector<int> res(nums.size(), -1);
        int a = 0, b = nums.size()-1;
        for (int i = 0; i < nums.size(); i++) {
            if (i%2==0) res[i] = nums[a++];
            else res[i] = nums[b--];
        }
        return res;
    }
};