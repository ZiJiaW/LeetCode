class Solution {
    // 贪心即可
public:
    int minimumBuckets(string street) {
        int res = 0;
        for (int i = 0; i < street.size(); i++) {
            if (street[i] == 'H') {
                if (i == 0 || street[i-1] == 'H') {
                    if (i == street.size()-1 || street[i+1] == 'H') return -1;
                    street[i+1] = '#';
                    res++;
                    continue;
                }
                if (street[i-1] == '#') continue;
                if (i == street.size()-1 || street[i+1] == 'H') {
                    if (street[i-1] != '.') return -1;
                    street[i-1] = '#';
                    res++;
                    continue;
                }
                street[i+1] = '#';
                res++;
            }
        }
        return res;
    }
};