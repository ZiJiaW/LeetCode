class Solution {
    // 先拓扑排序，然后按拓扑序计算到当前结点上的所有路径中的颜色树即可
public:
    int largestPathValue(string colors, vector<vector<int>>& edges) {
        int n = colors.size();
        vector<vector<int>> adj(n, vector<int>());
        for (auto &e : edges) {
            adj[e[0]].push_back(e[1]);
        }
        vector<int> v(n, -1), order;// topological order
        bool acyclic = true;
        function<void(int)> dfs = [&](int i) {
            if (v[i] == 1) return;
            if (v[i] == 0) {
                acyclic = false;
                return;
            }
            v[i] = 0;
            for (auto j : adj[i]) dfs(j);
            order.push_back(i);
            v[i] = 1;
        };
        for (int i = 0; i < n; i++) if (v[i] == -1) dfs(i);
        if (!acyclic) return -1;
        reverse(begin(order), end(order));
        vector<vector<int>> m(n, vector<int>(26, 0));
        int res = 0;
        for (auto k : order) {
            res = max(res, ++m[k][colors[k]-'a']);
            for (auto j : adj[k]) {
                for (int i = 0; i < 26; i++) {
                    m[j][i] = max(m[j][i], m[k][i]);
                }
            }
        }
        return res;
    }
};