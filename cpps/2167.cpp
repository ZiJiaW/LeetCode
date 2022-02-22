class Solution
{
public:
    int minimumTime(string s)
    {
        int res = INT_MAX, left = 0, size = s.size();
        for (int i = 0; i < size; i++)
        {
            if (s[i] == '1')
            {
                // 计算到i为止，左侧的最小cost
                left = min(left + 2, i + 1);
            }
            // 而后遍历，右侧全部直接消除
            res = min(res, left + size - i - 1);
        }
        return res;
    }
};