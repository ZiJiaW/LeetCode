class Solution
{
public:
    int flipgame(vector<int> &fronts, vector<int> &backs)
    {
        unordered_set<int> s;
        for (int i = 0; i < fronts.size(); i++)
        {
            if (fronts[i] == backs[i])
                s.insert(fronts[i]);
        }
        int res = INT_MAX;
        for (int i = 0; i < fronts.size(); i++)
        {
            if (s.find(fronts[i]) == end(s))
                res = min(res, fronts[i]);
            if (s.find(backs[i]) == end(s))
                res = min(res, backs[i]);
        }
        return res == INT_MAX ? 0 : res;
    }
};