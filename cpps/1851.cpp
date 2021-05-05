class Solution {
public:
    vector<int> minInterval(vector<vector<int>>& I, vector<int>& queries) {
        sort(begin(I), end(I), [](auto& a, auto& b) {
            return a[1] - a[0] < b[1] - b[0];
        });
        set<pair<int,int>> s;
        for (int i = 0; i < queries.size(); i++) {
            s.insert({queries[i], i});
        }
        vector<int> res(queries.size(), -1);
        // 从最小的区间开始，一个个把query拿掉，这样循环次数为m+n
        for (auto &v : I) {
            auto i = s.lower_bound({v[0], 0});
            while (i != end(s) && i->first <= v[1]) {
                res[i->second] = v[1] - v[0] + 1;
                i = s.erase(i);
            }
        }
        return res;
    }
};