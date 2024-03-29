class Solution {
    // 脑筋急转弯。。。
public:
    int minCost(vector<int>& startPos, vector<int>& homePos, vector<int>& rowCosts, vector<int>& colCosts) {
        int res = 0;
        if (startPos[0] < homePos[0]) {
            for (int i = startPos[0]+1; i <= homePos[0]; i++)
                res += rowCosts[i];
        } else if (startPos[0] > homePos[0]) {
            for (int i = startPos[0]-1; i >= homePos[0]; i--)
                res += rowCosts[i];
        }
        if (startPos[1] < homePos[1]) {
            for (int i = startPos[1]+1; i <= homePos[1]; i++) 
                res += colCosts[i];
        } else if (startPos[1] > homePos[1]) {
            for (int i = startPos[1]-1; i >= homePos[1]; i--) 
                res += colCosts[i];
        }
        return res;
    }
};