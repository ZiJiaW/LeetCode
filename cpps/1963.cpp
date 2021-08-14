class Solution {
    // remove already valid brackets
    // remaining string is like ]]][[[
    // it takes (len+1)/2 swaps to validate it
    // (just consider validate 2 pairs in one swap)
public:
    int minSwaps(string s) {
        int count = 0;
        for (auto c : s) {
            if (c == '[') count++;
            if (c == ']' && count > 0) count--;
        }
        return (count+1)/2;
    }
};