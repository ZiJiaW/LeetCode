class Solution
{ // 常规的prefix sum类型问题
public:
    vector<int> platesBetweenCandles(string s, vector<vector<int>> &queries)
    {
        vector<int> lcandle(s.size(), -1), rcandle(s.size(), -1), count(s.size(), -1);
        int l = -1, r = -1;
        for (int i = 0; i < s.size(); i++)
        {
            if (s[i] == '*')
                count[i] = i == 0 ? 1 : 1 + count[i - 1];
            else
                count[i] = i == 0 ? 0 : count[i - 1];
            if (s[i] == '|')
                l = i;
            lcandle[i] = l;
            int j = s.size() - 1 - i;
            if (s[j] == '|')
                r = j;
            rcandle[j] = r;
        }
        vector<int> res;
        for (auto &&q : queries)
        {
            l = rcandle[q[0]];
            r = lcandle[q[1]];
            if (l == -1 || r == -1 || l >= r)
                res.push_back(0);
            else
            {
                res.push_back(count[r] - count[l]);
            }
        }
        return res;
    }
};