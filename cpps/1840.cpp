class Solution {
    // 先两个方向上调整最大高度限制，使其符合斜率为1，这样最大高度都能取到
    // 而后就可以简单地找公式算出中间的最大高度
public:
    int maxBuilding(int n, vector<vector<int>>& r) {
        if (r.size() == 0) return n-1;
        r.push_back({1, 0});
        r.push_back({n, n-1});
        sort(begin(r), end(r), [](auto& a, auto& b) {
            return a[0] < b[0];
        });
        for (int i = 1; i < r.size(); i++) {
            r[i][1] = min(r[i][1], r[i-1][1] + r[i][0] - r[i-1][0]);
        }
        for (int i = r.size() - 2; i >= 0; i--) {
            r[i][1] = min(r[i][1], r[i+1][1] + r[i+1][0] - r[i][0]);
        }
        int res = 0;
        for (int i = 1; i < r.size(); i++) {
            res = max(res, (r[i][0] - r[i-1][0] + r[i][1] + r[i-1][1]) / 2);
        }
        return res;
    }
};