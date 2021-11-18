class Solution {
public:
    // 简单的找规律题
    string decodeCiphertext(string encodedText, int rows) {
        int len = encodedText.size();
        int cols = len / rows;
        string res;
        for (int i = 0; i < cols; i++) {
            for (int j = i; j < len; j += cols+1)
                res.push_back(encodedText[j]);
        }
        for (int i = res.size() - 1; i >= 0; i--) {
            if (res[i] != ' ') {
                res.resize(i+1);
                break;
            }
        }
        return res;
    }
};