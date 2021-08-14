class Solution {
    // priority queue | easy
public:
    int minStoneSum(vector<int>& piles, int k) {
        priority_queue<int> q;
        int sum = 0;
        for (auto i : piles) {
            q.push(i);
            sum += i;
        }
        while (k-- > 0) {
            auto t = q.top();
            if (t <= 1) break;
            q.pop();
            sum -= t / 2;
            t -= t / 2;
            q.push(t);
        }
        return sum;
    }
};