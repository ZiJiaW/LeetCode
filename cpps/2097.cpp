class Solution {
    // 求欧拉迹的算法
    // 既然遇到了就复习一下咯
    // 从起点开始深搜，将走过的边删除，当无边可走时将当前位置输出
    // 输出的即为欧拉迹的倒置
    // 起点如何找寻：入度小于出度
public:
    vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
        vector<int> path;
        unordered_map<int, vector<int>> adj;
        unordered_map<int, int> indeg;
        for (auto &p : pairs) {
            adj[p[0]].push_back(p[1]);
            indeg[p[0]];
            indeg[p[1]]++;
        }
        int s = pairs[0][0];
        for (auto [k, deg] : indeg) {
            if (deg < adj[k].size()) {
                s = k;
                break;
            }
        }
        function<void(int)> dfs = [&](int i) {
            while (!adj[i].empty()) {
                int j = adj[i].back();
                adj[i].pop_back();
                dfs(j);
            }
            path.push_back(i);
        };
        dfs(s);
        vector<vector<int>> res;
        for (int i = path.size() - 1; i > 0; i--) {
            res.push_back(vector<int>{path[i], path[i-1]});
        }
        return res;
    }
};