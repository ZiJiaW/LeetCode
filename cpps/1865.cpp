class FindSumPairs {
public:
    unordered_map<int, int> mp;
    vector<int> a, b;
    FindSumPairs(vector<int>& arr, vector<int>& brr) {
        a = arr;
        b = brr;
        for(auto x : b)
            mp[x] += 1;
    }
    
    void add(int index, int val) {
        mp[b[index]] -= 1;
        if(mp[b[index]] == 0)
            mp.erase(b[index]);
        b[index] += val;
        mp[b[index]] += 1;
    }
    
    int count(int val) {
        int ans = 0;
        for(auto x : a)
            if(mp.count(val - x))
                ans += mp[val - x];
    	return ans;
    }
};

/**
 * Your FindSumPairs object will be instantiated and called as such:
 * FindSumPairs* obj = new FindSumPairs(nums1, nums2);
 * obj->add(index,val);
 * int param_2 = obj->count(tot);
 */