class SeatManager {
    set<int> s;
    int cur;
public:
    SeatManager(int n): cur(0) {}
    
    int reserve() {
        if (s.empty()) {
            return ++cur;
        } else {
            int res = *begin(s);
            s.erase(begin(s));
            return res;
        }
    }
    
    void unreserve(int seatNumber) {
        s.insert(seatNumber);
    }
};

/**
 * Your SeatManager object will be instantiated and called as such:
 * SeatManager* obj = new SeatManager(n);
 * int param_1 = obj->reserve();
 * obj->unreserve(seatNumber);
 */