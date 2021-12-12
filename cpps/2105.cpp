class Solution
{
public:
    int minimumRefill(vector<int> &plants, int capacityA, int capacityB)
    {
        int res = 0;
        int i = 0, j = plants.size() - 1, ca = capacityA, cb = capacityB;
        while (i < j)
        {
            if (plants[i] <= ca)
                ca -= plants[i];
            else
            {
                res++;
                ca = capacityA - plants[i];
            }
            if (plants[j] <= cb)
                cb -= plants[j];
            else
            {
                res++;
                cb = capacityB - plants[j];
            }
            i++;
            j--;
        }
        if (i == j && max(ca, cb) < plants[i])
        {
            res++;
        }
        return res;
    }
};