class Solution {
public:
    vector<double> getCollisionTimes(vector<vector<int>>& cars) {
        int n = cars.size();
        stack<int> s;
        vector<double> res(n, -1);
        for (int i = n-1; i >= 0; i--) {
            int p1 = cars[i][0], s1 = cars[i][1];
            while (!s.empty()) {
                int j = s.top();
                int p2 = cars[j][0], s2 = cars[j][1];
                if (s1 <= s2 || (res[j] > 0 && 1.0*(p2-p1)/(s1-s2) >= res[j])) s.pop();
                else break;
            }
            if (!s.empty()) {
                res[i] = 1.0*(cars[s.top()][0]-p1)/(s1-cars[s.top()][1]);
            }
            s.push(i);
        }
        return res;
    }
};