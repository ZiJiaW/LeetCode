class Solution {
    // BFS为高度赋值即可
public:
    vector<vector<int>> highestPeak(vector<vector<int>>& isWater) {
        int m = isWater.size(), n = isWater[0].size();
        queue<int> q;
        vector<vector<int>> height(m, vector<int>(n, -1));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isWater[i][j] == 1) {
                    height[i][j] = 0;
                    q.push(i*n+j);
                }
            }
        }
        while (!q.empty()) {
            int t = q.front();
            q.pop();
            int i = t/n, j = t%n, h = height[i][j];
            if (i-1 >= 0 && height[i-1][j] == -1) {
                height[i-1][j] = h + 1;
                q.push((i-1)*n+j);
            }
            if (j-1 >= 0 && height[i][j-1] == -1) {
                height[i][j-1] = h + 1;
                q.push(i*n+j-1);
            }
            if (i+1 < m && height[i+1][j] == -1) {
                height[i+1][j] = h + 1;
                q.push((i+1)*n+j);
            }
            if (j+1 < n && height[i][j+1] == -1) {
                height[i][j+1] = h + 1;
                q.push(i*n+j+1);
            }
        }
        return height;
    }
};