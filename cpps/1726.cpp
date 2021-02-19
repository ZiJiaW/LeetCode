class Solution {
    /*
    traverse once to find the count of same-production pairs;
    multiply it by 8;
    */
public:
    int tupleSameProduct(vector<int>& nums) {
        unordered_map<int, int> m;
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = i+1; j < nums.size(); j++) {
                m[nums[i]*nums[j]]++;
            }
        }
        for (auto [_, v] : m) {
            res += v*(v-1)/2;
        }
        return res*8;
    }
};