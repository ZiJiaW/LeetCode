class Solution {
    // 因为只有10个字符，记录下当前各字符的奇偶状态仅1024种。
    // 用数组就可以记录出现过的状态数量
    // 我们计算与当前[0:i]奇偶状态仅差1位的[0:j](j<i)的次数之和。
public:
    long long wonderfulSubstrings(string word) {
        vector<int> count(1<<10, 0);
        count[0] = 1;
        int mask = 0;
        long long res = 0;
        for (auto c : word) {
            mask ^= 1 << (c - 'a');
            res += count[mask];
            for (int k = 0; k < 10; k++) {
                res += count[mask ^ (1 << k)];
            }
            count[mask]++;
        }
        return res;
    }
};