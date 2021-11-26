class RangeFreqQuery {
    // 存储区间，二分查找
    unordered_map<int, vector<tuple<int,int>>> m;
    int find_count(const vector<tuple<int,int>>& v, int idx) {
        int lo = 0, hi = v.size() - 1;
        while (lo <= hi) {
            int mid = (lo+hi)/2;
            auto [l, r] = v[mid];
            if (l <= idx && r > idx) return mid;
            else if (idx < l) hi = mid-1;
            else lo = mid+1;
        }
        return v.size();
    }
public:
    RangeFreqQuery(vector<int>& arr) {
        for (int i = 0; i < arr.size(); i++) {
            int prev_idx = m[arr[i]].empty() ? -1 : get<1>(m[arr[i]].back());
            m[arr[i]].push_back(make_tuple(prev_idx, i));
        }
    }
    
    int query(int left, int right, int value) {
        if (m.find(value) == m.end()) return 0;
        auto& t = m[value];
        return find_count(t, right) - find_count(t, left-1);
    }
};

/**
 * Your RangeFreqQuery object will be instantiated and called as such:
 * RangeFreqQuery* obj = new RangeFreqQuery(arr);
 * int param_1 = obj->query(left,right,value);
 */