class Solution {
public:
    int findCenter(vector<vector<int>>& edges) {
        int n = edges.size() + 1;
        vector<int> deg(n+1, 0);
        for (auto& e : edges) {
            deg[e[0]]++;
            deg[e[1]]++;
        }
        for (int i = 1; i <= n; i++) if (deg[i] == n-1) return i;
        return -1;
    }
};