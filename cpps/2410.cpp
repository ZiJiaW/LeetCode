class Solution {
    // 排序 贪心即可
public:
    int matchPlayersAndTrainers(vector<int>& players, vector<int>& trainers)
    {
        sort(begin(players), end(players));
        sort(begin(trainers), end(trainers));
        int i = 0;
        int res = 0;
        for (auto x : players) {
            while (i < trainers.size() && trainers[i] < x)
                i++;
            if (i == trainers.size())
                break;
            res++;
            i++;
        }
        return res;
    }
};