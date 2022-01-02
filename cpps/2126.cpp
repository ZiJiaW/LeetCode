class Solution
{
public:
    bool asteroidsDestroyed(int mass, vector<int> &asteroids)
    {
        sort(begin(asteroids), end(asteroids));
        int i = 0;
        int64_t sum = mass;
        while (i < asteroids.size() && asteroids[i] <= sum)
        {
            sum += asteroids[i++];
        }
        return i == asteroids.size();
    }
};