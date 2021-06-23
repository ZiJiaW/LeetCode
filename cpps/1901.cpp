class Solution {
    // 获取中间列/行的最大值，判断其左右/上下元素的大小，寻找山峰
public:
    vector<int> findPeakGrid(vector<vector<int>>& mat) {
        auto m = mat.size(), n = mat[0].size();
        vector<int> res{-1, -1};
        function<void(int,int)> conq = [&](int l, int r) {
            if (l == r) return;
            int mid = l + (r - l) / 2;
            int maxElem = -1, maxRow = -1;
            for (int i = 0; i < m; i++) {
                if (mat[i][mid] > maxElem) {
                    maxElem = mat[i][mid];
                    maxRow = i;
                }
            }
            if ((mid > 0 && mat[maxRow][mid] > mat[maxRow][mid-1] || mid == 0) &&
               (mid + 1 < n && mat[maxRow][mid] > mat[maxRow][mid+1] || mid + 1 == n)) {
                res[0] = maxRow;
                res[1] = mid;
                return;
            }
            if (mid > 0 && mat[maxRow][mid] < mat[maxRow][mid-1]) conq(l, mid);
            else if (mid + 1 < n && mat[maxRow][mid] < mat[maxRow][mid+1]) conq(mid+1, r);
        };
        conq(0, n);
        return res;
    }
};