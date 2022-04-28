struct FenwickTree
{
    FenwickTree(int n) : size(n), bit(n, 0) {}

    FenwickTree(vector<int> a) : FenwickTree(a.size())
    {
        for (size_t i = 0; i < a.size(); i++)
            add(i, a[i]);
    }

    int sum(int r)
    {
        int ret = 0;
        for (; r >= 0; r = (r & (r + 1)) - 1)
            ret += bit[r];
        return ret;
    }

    int range_sum(int l, int r)
    {
        return sum(r) - sum(l - 1);
    }

    void add(int idx, int delta)
    {
        for (; idx < size; idx = idx | (idx + 1))
            bit[idx] += delta;
    }

private:
    int size;
    vector<int> bit;
};

class Solution
{
public:
    long long goodTriplets(vector<int> &nums1, vector<int> &nums2)
    {
        int n = nums1.size();
        vector<int> pos(n, 0); // pos[i] means position of i in nums2
        for (int i = 0; i < n; i++)
        {
            pos[nums2[i]] = i;
        }
        vector<int> left(n, 0), right(n, 0);
        FenwickTree p(n);
        long long res = 0;
        for (int i = 1; i < n - 1; i++)
        {
            p.add(pos[nums1[i - 1]], 1);
            left[i] = p.sum(pos[nums1[i]] - 1);
            long long right = n - i - 1 - (pos[nums1[i]] - left[i]); // 右侧的可以直接减出来，等于nums1中右侧数量减去(nums2中左侧非重合的数量)
            res += right * left[i];
        }
        return res;
    }
};