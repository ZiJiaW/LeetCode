class Solution {
    // 暴力深搜解决
public:
    int maximalPathQuality(vector<int>& values, vector<vector<int>>& edges, int maxTime) {
        int n = values.size();
        vector<vector<pair<int,int>>> adj(n, vector<pair<int,int>>{});
        vector<int> v(n, 0);
        for (auto& e : edges) {
            adj[e[0]].push_back(make_pair(e[1], e[2]));
            adj[e[1]].push_back(make_pair(e[0], e[2]));
        }
        int res = 0;
        function<void(int,int,int)> dfs = [&](int i, int t, int s) {
            if (t > maxTime) return;
            if (i == 0) res = max(res, s);
            v[i]++;
            for (auto [j, d] : adj[i]) {
                dfs(j, t+d, s+(v[j]>0 ? 0 : values[j]));
            }
            v[i]--;
        };
        dfs(0, 0, values[0]);
        return res;
    }
};