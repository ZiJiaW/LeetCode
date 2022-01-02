class Solution
{
public:
    int numberOfBeams(vector<string> &bank)
    {
        int res = 0, prev = 0;
        for (auto &&s : bank)
        {
            int count = 0;
            for (auto c : s)
                count += c == '1';
            if (count > 0)
            {
                res += prev * count;
                prev = count;
            }
        }
        return res;
    }
};