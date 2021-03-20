class Solution {
public:
    int beautySum(string s) {
        vector<vector<int>> m(26, vector<int>(s.size(), 0));
        m[s[0]-'a'][0] = 1;
        for (int j = 1; j < s.size(); j++) {
            for (int i = 0; i < 26; i++) m[i][j] = m[i][j-1];
            m[s[j]-'a'][j]++;
        }
        int res = 0, tmp = 0;
        for (int i = 0; i < s.size(); i++) {
            for (int j = i+2; j < s.size(); j++) {
                int min_ = INT_MAX;
                int max_ = INT_MIN;
                for (int k = 0; k < 26; k++) {
                    tmp = i == 0 ? m[k][j] : m[k][j] - m[k][i-1];
                    if (tmp == 0) continue;
                    min_ = min(min_, tmp);
                    max_ = max(max_, tmp);
                }
                if (max_ > min_) res += max_ - min_;
            }
        }
        return res;
    }
};