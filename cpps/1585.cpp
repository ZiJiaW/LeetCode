class Solution {
    /*
    题目的本质是，对于t的相同位置，能否从s中找到一个相同的数字移过去，这个移动总是从后往前的（因为对t从前往后寻找）
    移动的规则：较小的数字能够和前面相邻的较大数字交换位置
    当成功移动过去后，我们就着眼于t的下一位，用idx存储对应数字在s中的下一个位置，这样下一次移动的时候从记录的位置开始查找即可。
    */
public:
    bool isTransformable(string s, string t) {
        vector<int> idx(10, -1);
        for (auto c : t) {
            bool found = false;
            for (int i = idx[c-'0']+1; i < s.size(); i++) {
                if (s[i] < c) return false;
                if (s[i] == c) {
                    idx[c-'0'] = i;// 记录找到的位置
                    s[i] += 10;// 把成功移动的数字变大，因为它实际上不会影响别的数字的移动了
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }
};