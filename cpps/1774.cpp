class Solution {
public:
    int closestCost(vector<int>& baseCosts, vector<int>& toppingCosts, int target) {
        int n = baseCosts.size(), m = toppingCosts.size();
        int res = 3e5;
        function<void(int, int)> dfs = [&](int i, int cost) {
            if (cost > target && abs(cost-target) >= abs(res-target)) return;
            if (abs(cost-target) < abs(res-target)) res = cost;
            if (abs(cost-target) == abs(res-target) && cost < res) res = cost;
            if (i == m) return;
            dfs(i+1, cost+2*toppingCosts[i]);
            dfs(i+1, cost+toppingCosts[i]);
            dfs(i+1, cost);
        };
        for (auto base : baseCosts) {
            dfs(0, base);
        }
        return res;
    }
};