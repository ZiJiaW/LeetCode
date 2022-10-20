class Solution {
public:
    vector<int> productQueries(int n, vector<vector<int>>& queries) {
        vector<int> arr;
        int i = 0;
        int64_t MOD = 1e9+7;
        int64_t m = (1 << 30) % MOD;
        while (n) {
            if (n % 2) arr.push_back(i);
            i++;
            n /= 2;
        }
        vector<int> res;
        res.reserve(queries.size());
        for (int i = 1; i < arr.size(); i++) {
            arr[i] = arr[i] + arr[i-1];
        }
        for (auto& q : queries) {
            int t = q[0] > 0 ? arr[q[1]] - arr[q[0]-1] : arr[q[1]];
            int64_t k = 1;
            while (t > 30) {
                k = k * m % MOD;
                t -= 30;
            }
            k *= 1 << t;
            k %= MOD;
            res.push_back(k);
        }
        return res;
    }
};