class Solution {
    // remove not-big-enough room from a set after sorting queries and room by size
public:
    vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries) {
        set<int> s;
        vector<int> res(queries.size(), -1);
        vector<int> qid;
        for (auto& room:rooms) {
            s.insert(room[0]);
        }
        for (int i = 0; i < queries.size(); i++) {
            qid.push_back(i);
        }
        sort(begin(qid), end(qid), [&](int a, int b) {
            return queries[a][1] < queries[b][1]; 
        });
        sort(begin(rooms), end(rooms), [](auto& a, auto& b) {
            return a[1] < b[1]; 
        });
        auto r = begin(rooms);
        for (int idx:qid) {
            auto& q = queries[idx];
            while (r != end(rooms) && (*r)[1] < q[1]) {
                s.erase((*r)[0]);
                r++;
            }
            if (r != end(rooms)) {
                auto it = s.upper_bound(q[0]);
                if (it == s.end()) {
                    res[idx] = *prev(it);
                } else if (it == s.begin()) {
                    res[idx] = *it;
                } else {
                    int l = abs(*prev(it) - q[0]);
                    int r = abs(*it - q[0]);
                    if (l <= r) res[idx] = *prev(it);
                    else res[idx] = *it;
                }
            }
        }
        return res;
    }
};