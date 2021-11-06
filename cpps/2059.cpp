class Solution {
    // BFS (start from start or goal)
public:
    int minimumOperations(vector<int>& nums, int start, int goal) {
        vector<bool> v(1001, false);
        v[start] = true;
        queue<int> q({start});
        int d = 0; bool found = false;
        while (!q.empty()) {
            int sz = q.size();
            while (--sz >= 0 && !found) {
                auto t = q.front(); q.pop();
                for (auto x : nums) {
                    if ((t+x) == goal || (t-x) == goal || (t^x) == goal) {
                        found = true;
                        break;
                    }
                    if (t+x >= 0 && t+x <= 1000 && !v[t+x]) {
                        v[t+x] = true;
                        q.push(t+x);
                    }
                    if (t-x >= 0 && t-x <= 1000 && !v[t-x]) {
                        v[t-x] = true;
                        q.push(t-x);
                    }
                    if ((t^x) >= 0 && (t^x) <= 1000 && !v[t^x]) {
                        v[t^x] = true;
                        q.push(t^x);
                    }
                }
            }
            d++;
            if (found) break;
        }
        return found ? d : -1;
    }
};