class Solution
{
public:
    int kIncreasing(vector<int> &arr, int k)
    {
        vector<int> dp;
        dp.reserve(arr.size() / k + 1);
        int res = 0;
        for (int i = 0; i < k; i++)
        {
            // i, i+k, i+2k......
            int len = 0;
            dp.clear();
            for (int j = i; j < arr.size(); j += k)
            {
                // 计算最长递增子序列
                auto it = upper_bound(begin(dp), end(dp), arr[j]);
                if (it == end(dp))
                {
                    dp.push_back(arr[j]);
                }
                else
                {
                    *it = arr[j];
                }
                len++;
            }
            res += len - dp.size();
        }
        return res;
    }
};