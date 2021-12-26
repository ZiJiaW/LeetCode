class Solution
{
public:
    vector<int> executeInstructions(int n, vector<int> &startPos, string s)
    {
        vector<int> res(s.size(), 0);
        int x = startPos[0], y = startPos[1];
        for (int i = 0; i < s.size(); i++)
        {
            int curx = x, cury = y, j = i;
            for (; j < s.size(); j++)
            {
                if (s[j] == 'L')
                    cury--;
                else if (s[j] == 'U')
                    curx--;
                else if (s[j] == 'R')
                    cury++;
                else
                    curx++;
                if (curx < 0 || cury < 0 || curx == n || cury == n)
                    break;
            }
            res[i] = j - i;
        }
        return res;
    }
};