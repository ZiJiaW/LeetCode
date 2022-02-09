class Solution
{
public:
    int minimumCost(vector<int> &cost)
    {
        sort(rbegin(cost), rend(cost));
        int res = 0;
        for (int i = 0; i < cost.size(); i += 3)
        {
            res += cost[i] + (i + 1 < cost.size() ? cost[i + 1] : 0);
        }
        return res;
    }
};