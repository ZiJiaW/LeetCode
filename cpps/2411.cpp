class Solution {
    // 按每一位记录下一个1所在的位置
    // 选择尽可能多的位为1的子序列长度，也就是取pos的最大值
public:
    vector<int> smallestSubarrays(vector<int>& nums)
    {
        vector<int> pos(30, 0);
        int n = nums.size();
        vector<int> res(n, 1);
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < 30; j++) {
                if (nums[i] & (1 << j))
                    pos[j] = i;
            }
            res[i] = max(1, *max_element(begin(pos), end(pos)) - i + 1);
        }
        return res;
    }
};