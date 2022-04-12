class Solution
{
public:
    vector<long long> maximumEvenSplit(long long finalSum)
    {
        if (finalSum & 1)
            return {};
        auto count = finalSum / 2;
        vector<long long> r;
        while (count > r.size())
        {
            r.push_back(2 * (r.size() + 1));
            count -= r.size();
        }
        if (count > 0)
            r.back() += 2 * count;
        return r;
    }
};