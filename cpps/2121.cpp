class Solution
{ // 用公式算每个位置的绝对值和 见https://leetcode.com/problems/intervals-between-identical-elements/discuss/1647567/c%2B%2B-or-O(n)-find-the-formula
public:
    vector<long long> getDistances(vector<int> &arr)
    {
        vector<long long> res(arr.size(), 0);
        unordered_map<int, vector<int>> m;
        for (int i = 0; i < arr.size(); i++)
        {
            m[arr[i]].push_back(i);
        }
        for (auto &&i : m)
        {
            auto &v = i.second;
            long long sum = 0;
            for (auto x : v)
                sum += x;
            long long n = v.size();
            for (int j = 0; j < v.size(); j++)
            {
                res[v[j]] = sum - (n - 2 * j) * v[j];
                sum -= 2 * v[j];
            }
        }
        return res;
    }
};