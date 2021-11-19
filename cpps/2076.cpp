class Solution {
    class ufs {
        vector<int> f;
        vector<int> rank;
    public:
        ufs(int size): f(size, 0), rank(size, 1) {
            for (int i = 0; i < size; i++) f[i] = i;
        }
        int find(int i) {
            if (f[i] == i) return i;
            f[i] = find(f[i]);
            return f[i];
        }
        void link(int i, int j) {
            int a = find(i), b = find(j);
            if (a == b) return;
            if (rank[a] < rank[b]) {
                f[a] = b;
            } else if (rank[a] > rank[b]) {
                f[b] = a;
            } else {
                f[a] = b;
                rank[b]++;
            }
        }
    };
public:
    // 暴力+并查集即可
    vector<bool> friendRequests(int n, vector<vector<int>>& restrictions, vector<vector<int>>& requests) {
        ufs s(n);
        vector<bool> res;
        for (auto& t : requests) {
            int t0 = s.find(t[0]), t1 = s.find(t[1]);
            bool flag = true;
            for (auto& r : restrictions) {
                int r0 = s.find(r[0]), r1 = s.find(r[1]);
                if ((r0 == t0 && r1 == t1) || (r0 == t1 && r1 == t0)) {
                    flag = false;
                    break;
                }
            }
            res.push_back(flag);
            if (flag) s.link(t0, t1);
        }
        return res;
    }
};