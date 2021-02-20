class Solution {
    // 计算总的xor值，然后与前偶数个的xor值比对即可得到最后一个
public:
    vector<int> decode(vector<int>& encoded) {
        int n = encoded.size() + 1, all = n, pre = 0;
        for (int i = 0; i < n-1; i += 2) {
            pre ^= encoded[i];
            all ^= (i+1)^(i+2);
        }
        vector<int> res(n, 0);
        res[n-1] = all^pre;
        for (int i = n-2; i >= 0; i--) res[i] = encoded[i] ^ res[i+1];
        return res;
    }
};