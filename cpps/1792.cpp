class Solution {
    /*
    选择获得最优增量的class插入；
    不过这个复杂度也太高了，必须预先计算好profit再插优先队列，否则TLE。
    */
public:
    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) {
        auto profit = [&](int i) {
            auto c = classes[i];
            return double(c[0]+1)/(c[1]+1) - double(c[0])/c[1];
        };
        priority_queue<pair<double, int>> q;
        double res = 0;
        for (int i = 0; i < classes.size(); i++) {
            res += double(classes[i][0])/classes[i][1];
            if (classes[i][0] == classes[i][1]) continue;
            q.push({profit(i), i});
        }
        if (q.empty()) return 1;
        while (extraStudents > 0) {
            auto [pf, i] = q.top();
            q.pop();
            extraStudents--;
            res += pf;
            classes[i][0]++;
            classes[i][1]++;
            q.push({profit(i), i});
        }
        return res/classes.size();
    }
};