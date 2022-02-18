class Solution
{
public:
    string subStrHash(string s, int power, int modulo, int k, int hashValue)
    {
        int64_t t = 0, pow_k = 1;
        reverse(begin(s), end(s));
        for (int i = 0; i < k; i++)
        {
            t = (t * power + s[i] - 'a' + 1) % modulo;
            if (i != k - 1)
                pow_k = pow_k * power % modulo;
        }
        int start = 0;
        for (int i = k; i <= s.size(); i++)
        {
            if (t == hashValue)
            {
                start = i - k;
            }
            if (i < s.size())
            {
                t -= static_cast<int64_t>(s[i - k] - 'a' + 1) * pow_k % modulo;
                if (t < 0)
                    t += modulo;
                t = (t * power + s[i] - 'a' + 1) % modulo;
            }
        }
        string r = s.substr(start, k);
        reverse(begin(r), end(r));
        return r;
    }
};