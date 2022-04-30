class Solution {
public:
    double champagneTower(int poured, int query_row, int query_glass)
    {
        double dp[100][100];
        dp[0][0] = poured;
        for (int i = 1; i <= query_row; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i)
                    dp[i][j] = dp[i - 1][0] > 1.0 ? (dp[i - 1][0] - 1) / 2.0 : 0;
                else {
                    double left = dp[i - 1][j - 1] > 1.0 ? (dp[i - 1][j - 1] - 1) / 2 : 0;
                    double right = dp[i - 1][j] > 1.0 ? (dp[i - 1][j] - 1) / 2 : 0;
                    dp[i][j] = left + right;
                }
            }
        }
        return dp[query_row][query_glass] >= 1 ? 1.0 : dp[query_row][query_glass];
    }
};