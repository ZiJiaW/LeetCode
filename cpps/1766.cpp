class Solution {
    int gcd(int m, int n) {
        if (m > n) return gcd(n, m);
        if (m == 0) return n;
        return gcd(n%m, m);
    }
public:
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        int n = nums.size();
        vector<vector<int>> adj(n, vector<int>{});
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        int g[51][51];
        for (int i = 1; i <= 50; i++) {
            for (int j = i; j <= 50; j++) {
                g[i][j] = g[j][i] = gcd(i, j);
            }
        }
        vector<int> res(n, -1), v(n, 0);
        vector<vector<tuple<int,int>>> appear(51, vector<tuple<int,int>>{});
        function<void(int,int)> dfs = [&](int i, int d) {
            v[i] = 1;
            int maxDepth = -1;
            for (int k = 1; k <= 50; k++) {
                if (appear[k].empty()) continue;
                if (g[k][nums[i]] == 1) {
                    auto [a, b] = appear[k].back();
                    if (b > maxDepth) {
                        maxDepth = b;
                        res[i] = a;
                    }
                }
            }
            appear[nums[i]].push_back({i, d});
            for (auto j : adj[i]) if (!v[j]) dfs(j, d+1);
            appear[nums[i]].pop_back();
        };
        dfs(0, 0);
        return res;
    }
};