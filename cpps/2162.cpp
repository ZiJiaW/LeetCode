class Solution
{
public:
    int minCostSetTime(int startAt, int moveCost, int pushCost, int targetSeconds)
    {
        int mm = 0, ss = targetSeconds, res = INT_MAX;
        while (ss >= 0)
        {
            if (ss < 100 && mm < 100)
            {
                int last = startAt;
                int cost = 0;
                int pre = 0;
                for (auto p : vector<int>{mm / 10, mm % 10, ss / 10, ss % 10})
                {
                    if (pre == 0 && p == 0)
                    {
                        continue;
                    }
                    if (p != 0)
                        pre++;
                    if (p != last)
                    {
                        cost += moveCost;
                        last = p;
                    }
                    cost += pushCost;
                }
                res = min(res, cost);
            }
            ss -= 60;
            mm++;
        }
        return res;
    }
};