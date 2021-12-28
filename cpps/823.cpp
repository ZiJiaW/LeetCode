class Solution
{
    // dp计算以某数为根的符合条件的二分树数量
public:
    int numFactoredBinaryTrees(vector<int> &arr)
    {
        int64_t mod = 1e9 + 7;
        int64_t res = 0;
        unordered_map<int, int64_t> m;
        for (auto n : arr)
            m[n] = 0;
        sort(begin(arr), end(arr));
        for (int i = 0; i < arr.size(); i++)
        {
            int64_t t = 1;
            for (int j = 0; j < i; j++)
            {
                if (arr[i] % arr[j] == 0 && m.find(arr[i] / arr[j]) != end(m))
                {
                    int l = arr[j], r = arr[i] / arr[j];
                    t += m[l] * m[r];
                    t %= mod;
                }
            }
            m[arr[i]] = t;
            res = (res + t) % mod;
        }
        return res;
    }
};