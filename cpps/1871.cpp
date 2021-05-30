class Solution {
    // 用dfs+dp数组的方式会TLE，用bfs更灵活
public:
    bool canReach(string s, int minJump, int maxJump) {
        if (s.back() == '1') return false;
        int n = s.size();
        queue<int> q({0});
        int midx = 0;
        while (!q.empty()) {
            auto t = q.front();
            q.pop();
            for (int j = max(midx, t+minJump); j <= min(t+maxJump, n-1); j++) {
                if (s[j] == '0') {
                    if (j == n - 1) return true;
                    q.push(j);
                }
            }
            midx = t+maxJump;
        }
        return false;
    }
};