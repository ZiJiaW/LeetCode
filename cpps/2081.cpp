class Solution {
    // 计算下一个回文数，然后判断再判断是否是k回文即可~
    // next parlindrome 写过n遍啦
public:
    long long kMirror(int k, int n) {
        deque<int> r{1};
        vector<int> tmp;
        auto calc = [&]() {
            long long res = 0;
            for (auto n : r) res = res*10 + n;
            return res;
        };
        auto get_next = [&]() {
            int i = (r.size()-1)/2; r[i]++;
            if (r[i] < 10 && r.size() % 2 == 1) return;
            while (r[i] == 10) {
                r[i] = 0;
                if (i == 0) {
                    r.push_front(1);
                    break;
                }
                r[i-1]++;
                i--;
            }
            int left = (r.size()-1)/2;
            int right = r.size()%2 == 0 ? left+1 : left;
            while (left >= 0) {
                r[right] = r[left];
                left--;  right++;
            }
        };
        auto check = [&](long long num) {
            tmp.clear();
            while (num) {
                tmp.push_back(num % k);
                num /= k;
            }
            for (int l = 0, r = tmp.size()-1; l < r; l++, r--) {
                if (tmp[l] != tmp[r]) return false;
            }
            return true;
        };
        long long sum = 0;
        while (n > 0) {
            long long s = calc();
            if (check(s)) {
                sum += s;
                n--;
            }
            get_next();
        }
        return sum;
    }
};