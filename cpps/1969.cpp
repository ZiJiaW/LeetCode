class Solution {
    int64_t mod = 1e9+7;
    int64_t pow(int64_t x, int64_t p) {
        if (p == 1) return x%mod;
        if (p == 0) return 1;
        int64_t r = pow(x, p/2);
        int64_t t = r*r%mod;
        if (p % 2 == 0) {
            return t % mod;
        } else {
            return t * (x % mod) % mod;
        }
    }
public:
    int minNonZeroProduct(int p) {
        // 0001 1110 1 14
        // 0010 1101 1 14
        // 0011 1100
        // 0100 1011
        // 0101 1010
        // 0110 1001
        // 0111 1000 1 14
        // 1111 15
        // (2^p-1) * (2^p-2)^(2^(p-1)-1)
        // just try making more 0x01
        int64_t t = 1;
        t <<= p;
        int64_t res = ((t-1)%mod) * (pow(t-2, t/2-1) % mod) % mod;
        return res;
    }
};