class SnapshotArray
{
    // 记录每个值的最小版本号，查询时二分查找
    struct item
    {
        int val;
        int ver;
    };
    vector<vector<item>> data_;
    int version_;

public:
    SnapshotArray(int length) : data_(length, vector<item>(1, item{0, 0})), version_(0) {}

    void set(int index, int val)
    {
        auto &items = data_[index];
        auto &last = items.back();
        if (last.ver == version_)
            last.val = val;
        else
            items.push_back(item{val, version_});
    }

    int snap()
    {
        return version_++;
    }

    int get(int index, int snap_id)
    {
        auto &items = data_[index];
        int lo = 0, hi = items.size();
        while (lo < hi - 1)
        {
            int mid = lo + (hi - lo) / 2;
            if (items[mid].ver <= snap_id)
                lo = mid;
            else
                hi = mid;
        }
        return items[lo].val;
    }
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */