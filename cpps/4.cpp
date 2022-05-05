class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
    {
        int m = nums1.size(), n = nums2.size();
        int lo = 0, hi = m, x, y, A_x, B_y, A_x_1, B_y_1;
        while (lo <= hi) {
            x = (lo + hi) / 2;
            y = (m + n + 1) / 2 - x;
            // x一定在有效范围内，但需要保证y的有效性
            if (y < 0) {
                hi = x - 1;
            } else if (y > n) {
                lo = x + 1;
            } else {
                A_x = x < m ? nums1[x] : INT_MAX;
                B_y = y < n ? nums2[y] : INT_MAX;
                A_x_1 = x > 0 ? nums1[x - 1] : INT_MIN;
                B_y_1 = y > 0 ? nums2[y - 1] : INT_MIN;
                if (A_x < B_y_1)
                    lo = x + 1;
                else if (B_y < A_x_1)
                    hi = x - 1;
                else {
                    if ((m + n) % 2 != 0) {
                        return max(A_x_1, B_y_1);
                    } else {
                        return (max(A_x_1, B_y_1) + min(A_x, B_y)) / 2.0;
                    }
                }
            }
        }
        return 0;
    }
};