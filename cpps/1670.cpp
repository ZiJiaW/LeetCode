class FrontMiddleBackQueue {
    list<int> left, right;
    // 维护两个链表，分别一半即可~~
public:
    FrontMiddleBackQueue() {}
    
    void move_left() {
        left.push_back(right.front());
        right.pop_front();
    }
    
    void move_right() {
        right.push_front(left.back());
        left.pop_back();
    }
    
    void pushFront(int val) {
        left.push_front(val);
        if (left.size() - right.size() == 2) move_right();
    }
    
    void pushMiddle(int val) {
        if (left.size() == right.size()) left.push_back(val);
        else {
            move_right();
            left.push_back(val);
        }
    }
    
    void pushBack(int val) {
        right.push_back(val);
        if (right.size() > left.size()) move_left();
    }
    
    int popFront() {
        if (left.empty()) return -1;
        int res = left.front();
        left.pop_front();
        if (left.size() < right.size()) move_left();
        return res;
    }
    
    int popMiddle() {
        if (left.empty()) return -1;
        int res = left.back();
        left.pop_back();
        if (left.size() < right.size()) move_left();
        return res;
    }
    
    int popBack() {
        if (right.empty() && left.size() == 1) {
            int res = left.front();
            left.pop_back();
            return res;
        }
        if (right.empty()) return -1;
        int res = right.back();
        right.pop_back();
        if (left.size() - right.size() == 2) move_right();
        return res;
    }
};

/**
 * Your FrontMiddleBackQueue object will be instantiated and called as such:
 * FrontMiddleBackQueue* obj = new FrontMiddleBackQueue();
 * obj->pushFront(val);
 * obj->pushMiddle(val);
 * obj->pushBack(val);
 * int param_4 = obj->popFront();
 * int param_5 = obj->popMiddle();
 * int param_6 = obj->popBack();
 */