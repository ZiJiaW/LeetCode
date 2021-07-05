class Solution {
    // 计算下一个回文数，然后判断素数，拼图题
    bool isPrime(int n) {
        if (n < 2) return false;
        for (int i = 2; i <= sqrt(n); i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
    bool isHalfLarge(string& s) {
        for (int i = s.size()/2 - 1; i >= 0; i--) {
            if (s[i] > s[s.size()-1-i]) return true;
            else if (s[i] < s[s.size()-1-i]) return false;
        }
        return false;
    }
    int nextPalindrome(int n) {
        if (n < 9) {
            return n+1;
        }
        if (n == 9) {
            return 11;
        }
        string s = to_string(n);
        string next;
        if (isHalfLarge(s)) {
            next = s.substr(0, (s.size()+1)/2);
        } else {
            string half = s.substr(0, (s.size()+1)/2);
            next = to_string(stoi(half)+1);
        }
        if (next.size() == (s.size()+1)/2) {
            for (int i = 0; i < next.size(); i++) {
                s[i] = next[i];
                s[s.size()-1-i] = s[i];
            }
            return stoi(s);
        } else {
            int nextLen = s.size() + 1;
            int t = 1;
            for (int i = 1; i < nextLen; i++) t *= 10;
            return nextPalindrome(t);
        }
    }
    
public:
    int primePalindrome(int n) {
        bool flag = isPrime(n);
        if (isPrime(n)) {
            string s = to_string(n);
            bool flag = true;
            for (int i = 0; i < s.size()/2; i++) {
                if (s[i] != s[s.size()-1-i]) {
                    flag = false;
                    break;
                }
            }
            if (flag) return n;
        }
        while (true) {
            n = nextPalindrome(n);
            if (isPrime(n)) return n;
        }
        return -1;
    }
};