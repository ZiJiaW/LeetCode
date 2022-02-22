class Solution
{
public:
    long long smallestNumber(long long num)
    {
        if (num == 0)
            return 0;
        bool neg = num < 0;
        string s = to_string(abs(num));
        if (neg)
        {
            sort(rbegin(s), rend(s));
            return -stoll(s);
        }
        else
        {
            sort(begin(s), end(s));
            int i = 0;
            while (s[i] == '0')
            {
                i++;
            }
            swap(s[0], s[i]);
            return stoll(s);
        }
    }
};