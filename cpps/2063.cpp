class Solution {
    // 简单的组合
public:
    long long countVowels(string word) {
        long long res = 0;
        for (int i = 0; i < word.size(); i++) {
            if (word[i] == 'a' || word[i] == 'e' || word[i] == 'i' || word[i] == 'o' || word[i] == 'u') {
                long long left = i, right = word.size() - i - 1;
                res += left + right + left * right + 1;
            }
        }
        return res;
    }
};