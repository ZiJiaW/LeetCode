class Solution
{
    // record locked ( and unlocked indeces
    // prefer to use closest locked ( to match a locked )
    // then closest unlocked index
    // finally we handle the remaining locked (
public:
    bool canBeValid(string s, string locked)
    {
        if (s.size() % 2 == 1)
            return false;
        vector<int> lock;
        deque<int> unlock;
        for (int i = 0; i < s.size(); i++)
        {
            if (locked[i] == '0')
            {
                unlock.push_back(i);
            }
            else if (s[i] == ')')
            {
                if (!lock.empty())
                {
                    lock.pop_back();
                }
                else if (!unlock.empty())
                {
                    unlock.pop_back();
                }
                else
                {
                    return false;
                }
            }
            else
            {
                lock.push_back(i);
            }
        }
        for (auto i : lock)
        {
            while (!unlock.empty() && unlock.front() <= i)
            {
                unlock.pop_front();
            }
            if (unlock.empty())
            {
                return false;
            }
            unlock.pop_front();
        }
        return true;
    }
};