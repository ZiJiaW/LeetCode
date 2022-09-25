class Solution {
    // 寻找一个花费最大的交易顺序
    // 1. 先进行会亏损的交易，再进行有收益的
    // 2. 亏损的交易中，先进行回款小的
    // 3. 有收益的交易中，先进行花费大的
public:
    long long minimumMoney(vector<vector<int>>& transactions)
    {
        sort(begin(transactions), end(transactions), [](auto& a, auto& b) {
            if (a[0] > a[1] && b[0] <= b[1])
                return true;
            if (a[0] < a[1] && b[0] >= b[1])
                return false;
            if (a[0] > a[1])
                return a[1] < b[1];
            else
                return a[0] > b[0];
        });
        long long res = 0, m = 0;
        for (auto& v : transactions) {
            if (m < v[0]) {
                res += v[0] - m;
                m = 0;
            } else {
                m -= v[0];
            }
            m += v[1];
        }
        return res;
    }
};