class Solution {
    void next(string& s) {
        for (int i = s.size() - 2; i >= 0; i--) {
            if (s[i] < s[i+1]) {
                for (int j = s.size() - 1; j > i; j--) {
                    if (s[j] > s[i]) {
                        swap(s[i], s[j]);
                        reverse(begin(s) + i+1, end(s));
                        return;
                    }
                }
            }
        }
    }
    int distance(string& a, string& b) {
        int res = 0;
        for (int i = 0; i < a.size(); i++) {
            if (a[i] == b[i]) continue;
            for (int j = i+1; j < a.size(); j++) {
                if (a[j] == b[i]) {
                    res += j - i;
                    while (j > i) {
                        swap(a[j], a[j-1]);
                        j--;
                    }
                    break;
                }
            }
        }
        return res;
    }
public:
    int getMinSwaps(string num, int k) {
        string origin(num);
        for (int i = 0; i < k; i++) {
            next(num);
        }
        return distance(origin, num);
    }
};