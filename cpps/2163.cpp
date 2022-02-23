class Solution
{
    // 先计算左侧i(n~2n)个元素中最小n个元素的和，用left保存，用最小堆计算
    // 同理，再计算右侧i(n~2n)个元素中最大n个元素的和，计算过程中算左侧3n-i个元素最小n个和与其之差，记录其中最小的即可
    // 复杂度nlgn
public:
    long long minimumDifference(vector<int> &nums)
    {
        long long res = INT_MAX, t = 0, t2 = 0;
        priority_queue<int> pq;
        priority_queue<int, vector<int>, greater<int>> pq2;
        int n = nums.size() / 3;
        for (int i = 0; i < n; i++)
        {
            pq.push(nums[i]);
            pq2.push(nums[3 * n - i - 1]);
            t += nums[i];
            t2 += nums[3 * n - i - 1];
        }
        vector<long long> left(n + 1, 0);
        left[0] = t;
        for (int i = n; i < 2 * n; i++)
        {
            pq.push(nums[i]);
            auto rmv = pq.top();
            pq.pop();
            t = t - rmv + nums[i];
            left[i - n + 1] = t;
        }
        res = left[n] - t2;
        for (int i = 2 * n - 1; i >= n; i--)
        {
            pq2.push(nums[i]);
            auto rmv = pq2.top();
            pq2.pop();
            t2 = t2 - rmv + nums[i];
            res = min(res, left[i - n] - t2);
        }
        return res;
    }
};