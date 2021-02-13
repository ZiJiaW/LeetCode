/**
相连的位置构成图，把连通分量内的值一一比较即可。
*/
class Solution {
public:
    int minimumHammingDistance(vector<int>& source, vector<int>& target, vector<vector<int>>& allowedSwaps) {
        int n = source.size();
        vector<vector<int>> adj(n, vector<int>{});
        for (auto& e : allowedSwaps) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        int res = n;
        vector<int> s, t;
        vector<bool> v(n, false);
        function<void(int)> dfs = [&](int i) {
            v[i] = true;
            s.push_back(source[i]);
            t.push_back(target[i]);
            for (auto u : adj[i]) if (!v[u]) dfs(u);
        };
        for (int i = 0; i < n; i++) {
            if (!v[i]) {
                s.clear();
                t.clear();
                dfs(i);
                sort(begin(s), end(s));
                sort(begin(t), end(t));
                for (int k = 0, l = 0; k < s.size() && l < t.size();) {
                    if (s[k] == t[l]) {
                        res--;
                        k++; l++;
                    } else if (s[k] > t[l]) {
                        l++;
                    } else {
                        k++;
                    }
                }
            }
        }
        return res;
    }
};