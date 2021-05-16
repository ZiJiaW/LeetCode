class Solution {
public:
    int rearrangeSticks(int n, int k) {
        // 换个角度考虑1的位置，而非n的位置
        // f(n,k)=f(n-1,k-1)+(n-1)*f(n-1,k)
        vector<vector<int64_t>> dp(n+1, vector<int64_t>(k+1, 0));
        int64_t mod = 1e9 + 7;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k; j++) {
                if (i == j) dp[i][j] = 1;
                else if (i < j) dp[i][j] = 0;
                else {
                    dp[i][j] = (dp[i-1][j-1] + (i-1)*dp[i-1][j]) % mod;
                }
            }
        }
        return dp[n][k];
    }
};