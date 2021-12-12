class Solution
{
    // 前缀和计算，然后找到可以触及的区间即可
public:
    int maxTotalFruits(vector<vector<int>> &fruits, int startPos, int k)
    {
        int res = 0, maxPos = fruits.back()[0], j = 0;
        vector<int> sums(maxPos + 1, 0);
        for (int i = 0; i < sums.size(); i++)
        {
            if (j < fruits.size() && i == fruits[j][0])
                sums[i] += fruits[j++][1];
            if (i > 0)
                sums[i] += sums[i - 1];
        }
        for (int i = 0; i <= k; i++)
        {
            int r = startPos + i;                          // head right i steps
            int l = min(startPos - (k - 2 * i), startPos); // head left k-2i steps
            r = min(r, maxPos);
            l = max(l, 0);
            if (l > r)
                continue;
            res = max(res, l == 0 ? sums[r] : sums[r] - sums[l - 1]);
        }
        for (int i = 0; i <= k; i++)
        {
            int l = startPos - i;                          // head left i steps
            int r = max(startPos + (k - 2 * i), startPos); // head right k-2i steps
            r = min(r, maxPos);
            l = max(l, 0);
            if (l > r)
                continue;
            res = max(res, l == 0 ? sums[r] : sums[r] - sums[l - 1]);
        }
        return res;
    }
};