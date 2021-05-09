class Solution {
public:
    int maximumElementAfterDecrementingAndRearranging(vector<int>& arr) {
        sort(begin(arr), end(arr));
        arr[0] = 1;
        int res = 1;
        for (int i = 1; i < arr.size(); i++) {
            if (arr[i] > arr[i-1]) arr[i] = arr[i-1] + 1;
            res = max(res, arr[i]);
        }
        return res;
    }
};