class Solution {
public:
    int minSwaps(string s) {
        int count0 = 0, count1 = 0;
        for (auto c : s) {
            if (c == '1') count1++;
            else count0++;
        }
        if (abs(count0 - count1) > 1) return -1;
        vector<string> cands;
        if (count0 == count1) {
            string ss;
            for (int i = 0; i < s.size(); i++) {
                if (i&1) ss += '0';
                else ss += '1';
            }
            cands.push_back(ss);
            reverse(begin(ss), end(ss));
            cands.push_back(ss);
        } else if (count0 > count1) {
            string ss;
            for (int i = 0; i < s.size(); i++) {
                if (i&1) ss += '1';
                else ss += '0';
            }
            cands.push_back(ss);
        } else if (count0 < count1) {
            string ss;
            for (int i = 0; i < s.size(); i++) {
                if (i&1) ss += '0';
                else ss += '1';
            }
            cands.push_back(ss);
        }
        int res = INT_MAX;
        for (auto& cand : cands) {
            //cout << cand << endl;
            int r = 0;
            for (int i = 0; i < s.size(); i++) {
                if (s[i] != cand[i]) r++;
            }
            res = min(res, r/2);
        }
        return res;
    }
};