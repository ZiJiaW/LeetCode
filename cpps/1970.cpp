class Solution {
    // 并查集，从后往前倒放直到第一行与最后一行连通
    // 添加两个虚拟结点start和end用于快速计算连通性
public:
    class ufs {
    public:
        vector<int> f;
        vector<int> h;
        ufs(int len): f(len, -1), h(len, 0) {
            for (int i = 0; i < len; i++) f[i] = i;
        }
        int find(int x) {
            if (f[x] == x) return x;
            int t = find(f[x]);
            f[x] = t;
            return t;
        }
        void uni(int x, int y) {
            int xf = find(x);
            int yf = find(y);
            if (xf == yf) return;
            if (h[xf] == h[yf]) {
                f[xf] = yf;
                h[yf]++;
            } else if (h[xf] > h[yf]) {
                f[yf] = xf;
            } else {
                f[xf] = yf;
            }
        }
    };
    int latestDayToCross(int row, int col, vector<vector<int>>& cells) {
        ufs u(row*col+2);
        vector<bool> land(row*col+2, false);
        int start = row*col, end = row*col+1;
        for (int i = 0; i < col; i++) {
            u.uni(i, start);
            u.uni(col*(row-1)+i, end);
        }
        for (int i = cells.size() - 1; i >= 0; i--) {
            int x = cells[i][0] - 1;
            int y = cells[i][1] - 1;
            int cur = x*col+y;
            land[cur] = true;
            if (x > 0) {
                int left = (x-1)*col+y;
                if (land[left] == true) u.uni(cur, left);
            }
            if (y > 0) {
                int up = x*col+y-1;
                if (land[up] == true) u.uni(cur, up);
            }
            if (x < row-1) {
                int down = (x+1)*col+y;
                if (land[down] == true) u.uni(cur, down);
            }
            if (y < col - 1) {
                int right = x*col+y+1;
                if (land[right] == true) u.uni(cur, right);
            }
            if (u.find(start) == u.find(end)) {
                return i;
            }
        }
        return -1;
    }
};