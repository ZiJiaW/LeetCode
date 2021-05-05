class Solution {
public:
    bool splitString(string s) {
        function<bool(int,int64_t)> dfs = [&](int i, int64_t k) {
            if (i == s.size()) return true;
            if (k == 0) return false;
            if (k == -1) {
                for (int j = 1; j != s.size(); j++) {
                    auto cur = stoll(s.substr(i, j - i));
                    if (cur > 1e10) break;
                    if (dfs(j, cur))
                        return true;
                }
                return false;
            } else {
                for (int j = i+1; j != s.size() + 1; j++) {
                    auto cur = stoll(s.substr(i, j - i));
                    if (cur > 1e10 || cur > k - 1) break;
                    if (cur != k-1) continue;
                    if (dfs(j, cur))
                        return true;
                }
                return false;
            }
        };
        return dfs(0, -1);
    }
};