class Solution
{
public:
    int countElements(vector<int> &nums)
    {
        int nmax = INT_MIN, nmin = INT_MAX;
        for (auto n : nums)
        {
            nmax = max(nmax, n);
            nmin = min(nmin, n);
        }
        int res = 0;
        for (auto n : nums)
        {
            if (n > nmin && n < nmax)
            {
                res++;
            }
        }
        return res;
    }
};