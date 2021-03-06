class Solution {
public:
    int countMatches(vector<vector<string>>& items, string ruleKey, string ruleValue) {
        int res = 0;
        for (auto &item : items) {
            if (ruleKey == "type" && item[0] == ruleValue ||
                ruleKey == "color" && item[1] == ruleValue ||
                ruleKey == "name" && item[2] == ruleValue)
                res++;
        }
        return res;
    }
};