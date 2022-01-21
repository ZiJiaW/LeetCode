class Solution
{
    // 用26位记录出现的字母
public:
    int wordCount(vector<string> &startWords, vector<string> &targetWords)
    {
        unordered_set<int> s;
        auto to_mask = [](const string &s)
        {
            int tmp = 0;
            for (auto c : s)
            {
                tmp |= 1 << (c - 'a');
            }
            return tmp;
        };
        for (const auto &w : startWords)
        {
            s.insert(to_mask(w));
        }
        int res = 0;
        for (const auto &w : targetWords)
        {
            int mask = to_mask(w);
            for (int t = 1; t < (1 << 26); t <<= 1)
            {
                if (mask & t)
                {
                    if (s.find(mask & (~t)) != s.end())
                    {
                        res++;
                        break;
                    }
                }
            }
        }
        return res;
    }
};