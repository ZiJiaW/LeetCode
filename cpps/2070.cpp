class Solution {
    // 排序，维护当前price下最漂亮的值即可
public:
    vector<int> maximumBeauty(vector<vector<int>>& items, vector<int>& queries) {
        vector<tuple<int,int>> q;
        for (int i = 0; i < queries.size(); i++) q.push_back({queries[i], i});
        sort(begin(q), end(q), [](auto& x, auto& y) {
            return get<0>(x) < get<0>(y);
        });
        sort(begin(items), end(items), [](auto& x, auto& y) {
            return x[0] < y[0];
        });
        vector<int> res(queries.size(), 0);
        int i = 0, beauty = 0;
        for (auto& t : q) {
            while (i < items.size() && items[i][0] <= get<0>(t)) {
                beauty = max(beauty, items[i][1]);
                i++;
            }
            res[get<1>(t)] = beauty;
        }
        return res;
    }
};