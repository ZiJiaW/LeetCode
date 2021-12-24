class Solution
{
public:
    string addSpaces(string s, vector<int> &spaces)
    {
        string res(s.size() + spaces.size(), '#');
        int c = 0;
        for (auto &&pos : spaces)
        {
            res[pos + c] = ' ';
            c++;
        }
        for (int i = 0, j = 0; i < s.size();)
        {
            if (res[j] == '#')
            {
                res[j] = s[i];
                i++;
            }
            j++;
        }
        return res;
    }
};