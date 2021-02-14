class Solution {
public:
    int countHomogenous(string s) {
        int64_t res = 0, n = 1, mod = 1e9+7;
        char last = s[0];
        s.push_back('#');
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == last) n++;
            else {
                last = s[i];
                res = (res + n*(n+1)/2) % mod;
                n = 1;
            }
        }
        return res;
    }
};