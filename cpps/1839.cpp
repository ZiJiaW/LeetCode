class Solution {
public:
    int longestBeautifulSubstring(string word) {
        if (word.size() < 5) return 0;
        int res = 0, start = 0;
        for (int i = 1; i < word.size(); i++) {
            if (word[i] >= word[i-1]) {
                if (word[i] == word[i-1] ||
                    word[i-1] == 'a' && word[i] == 'e' ||
                    word[i-1] == 'e' && word[i] == 'i' ||
                    word[i-1] == 'i' && word[i] == 'o' ||
                    word[i-1] == 'o' && word[i] == 'u') {
                    if (word[i] == 'u' && word[start] == 'a') {
                        res = max(res, i - start + 1);
                    }
                } else {
                    start = i;
                }
            } else {
                start = i;
            }
        }
        return res;
    }
};