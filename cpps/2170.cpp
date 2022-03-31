class Solution
{
public:
    int minimumOperations(vector<int> &nums)
    {
        unordered_map<int, int> odd, even;
        for (int i = 0; i < nums.size(); i++)
        {
            if (i & 1)
                odd[nums[i]]++;
            else
                even[nums[i]]++;
        }
        vector<pair<int, int>> odds(odd.begin(), odd.end());
        vector<pair<int, int>> evens(even.begin(), even.end());
        auto cmp = [](const auto &a, const auto &b)
        { return a.second > b.second; };
        sort(odds.begin(), odds.end(), cmp);
        sort(evens.begin(), evens.end(), cmp);
        odds.push_back({0, 0});
        evens.push_back({0, 0});
        auto i = odds.begin(), j = evens.begin();
        if (i->first != j->first)
        {
            return nums.size() - i->second - j->second;
        }
        else
        {
            return nums.size() - max(i->second + (j + 1)->second, (i + 1)->second + j->second);
        }
    }
};