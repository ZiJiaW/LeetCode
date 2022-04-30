class Solution {
public:
    vector<int> groupStrings(vector<string>& words)
    {
        unordered_map<int, int> m;
        for (auto& w : words) {
            m[accumulate(begin(w), end(w), 0, [](int m, char c) { return m | (1 << (c - 'a')); })]++;
        }
        int group_num = 0, group_size = 0;
        function<void(int, int&)> dfs = [&](int mask, int& count) {
            auto it = m.find(mask);
            if (it != m.end()) {
                count += it->second;
                m.erase(it);
                // possible add
                for (int i = 0; i < 26; i++) {
                    if (!(mask & (1 << i)))
                        dfs(mask | (1 << i), count);
                }
                // possible delete
                for (int i = 0; i < 26; i++) {
                    if (mask & (1 << i)) {
                        dfs(mask ^ (1 << i), count);
                        // replace here
                        int t = mask ^ (1 << i); // delete i
                        for (int j = 0; j < 26; j++) { // add another j
                            if (j != i && !(t & (1 << j))) {
                                dfs(t | (1 << j), count);
                            }
                        }
                    }
                }
            }
        };
        while (!m.empty()) {
            int count = 0;
            dfs(begin(m)->first, count);
            group_size = max(group_size, count);
            group_num++;
        }
        return { group_num, group_size };
    }
};