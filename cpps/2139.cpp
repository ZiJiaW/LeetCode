class Solution
{
public:
    int minMoves(int target, int maxDoubles)
    {
        int res = 0;
        while (target != 1)
        {
            if (target % 2 == 0 && maxDoubles > 0)
            {
                maxDoubles--;
                target /= 2;
                res++;
            }
            else if (maxDoubles == 0)
            {
                res += target - 1;
                target = 1;
            }
            else
            {
                target--;
                res++;
            }
        }
        return res;
    }
};