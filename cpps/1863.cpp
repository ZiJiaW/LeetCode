class Solution {
public:
    int subsetXORSum(vector<int>& nums) {
        vector<int> r{0};
        int res = 0;
        for (auto n : nums) {
            int rs = r.size();
            for (int i = 0; i < rs; i++) {
                r.push_back(n ^ r[i]);
                res += r[i] ^ n;
            }
        }
        return res;
    }
};