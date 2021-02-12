class Solution {
public:
    string largestMerge(string word1, string word2) {
        string res;
        int i = 0, j = 0;
        while (true) {
            if (word1[i] > word2[j]) {
                res += word1[i++];
            } else if (word1[i] < word2[j]) {
                res += word2[j++];
            } else {
                int ti = i, tj = j;
                while (ti != word1.size() && tj != word2.size() && word1[ti] == word2[tj]) {
                    ti++; tj++;
                }
                if (ti == word1.size() && tj == word2.size()) {
                    res += word1[i++];
                } else if (ti == word1.size() || (tj != word2.size() && word1[ti] < word2[tj])) {
                    res += word2[j++];
                } else if (tj == word2.size() || (ti != word1.size() && word1[ti] > word2[tj])) {
                    res += word1[i++];
                }
            }
            if (i == word1.size())
                return res + word2.substr(j);
            if (j == word2.size())
                return res + word1.substr(i);
        }
        return res;
    }
};