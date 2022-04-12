class Solution
{
public:
    vector<long long> sumOfThree(long long num)
    {
        auto r = (num - 3) / 3;
        auto t = (num - 3) % 3;
        if (t != 0)
            return {};
        else
            return {r, r + 1, r + 2};
    }
};