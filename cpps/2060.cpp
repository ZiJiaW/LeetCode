class Solution {
    // 本题计算的是可行性，因此只要dfs中单个状态计算过了就可以不再访问了（不可行），否则找到解早就返回了。
public:
    bool possiblyEquals(string s1, string s2) {
        vector<vector<vector<bool>>> m(s1.size()+1, vector<vector<bool>>(s2.size()+1, vector<bool>(2001, false)));
        auto isdigit = [](char c) { return c >= '0' && c <= '9'; };
        function<bool(int,int,int)> dfs = [&](int i, int j, int d) {
            if (m[i][j][d+999]) return false;
            if (i == s1.size() && j == s2.size() && d == 0) return true;
            m[i][j][d+999] = true;
            if (d > 0) {
                if (j >= s2.size()) return false;
                else if (!isdigit(s2[j])) return dfs(i, j+1, d-1);
                else {
                    int num = 0;
                    for (int k = j; k < s2.size() && isdigit(s2[k]); k++) {
                        num = 10*num + s2[k] - '0';
                        if (dfs(i, k+1, d-num)) return true;
                    }
                }
            } else if (d < 0) {
                if (i >= s1.size()) return false;
                else if (!isdigit(s1[i])) return dfs(i+1, j, d+1);
                else {
                    int num = 0;
                    for (int k = i; k < s1.size() && isdigit(s1[k]); k++) {
                        num = 10*num + s1[k] - '0';
                        if (dfs(k+1, j, d+num)) return true;
                    }
                }
            } else {
                if (isdigit(s1[i])) {
                    int num = 0;
                    for (int k = i; k < s1.size() && isdigit(s1[k]); k++) {
                        num = 10*num + s1[k] - '0';
                        if (dfs(k+1, j, num)) return true;
                    }
                } else if (isdigit(s2[j])) {
                    int num = 0;
                    for (int k = j; k < s2.size() && isdigit(s2[k]); k++) {
                        num = 10*num + s2[k] - '0';
                        if (dfs(i, k+1, -num)) return true;
                    }
                } else {
                    if (s1[i] == s2[j]) return dfs(i+1, j+1, 0);
                    else return false;
                }
            }
            return false;
        };
        return dfs(0, 0, 0);
    }
};