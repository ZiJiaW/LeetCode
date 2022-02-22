class Bitset
{
    vector<int> b;
    int sz;
    int ones;

public:
    Bitset(int size) : b((size - 1) / 32 + 1, 0), sz(size), ones(0) {}

    void fix(int idx)
    {
        int i = idx / 32, j = idx % 32;
        if (!(b[i] & (1 << j)))
            ones++;
        b[i] |= 1 << j;
    }

    void unfix(int idx)
    {
        int i = idx / 32, j = idx % 32;
        if (b[i] & (1 << j))
            ones--;
        b[i] &= ~(1 << j);
    }

    void flip()
    {
        for (auto &t : b)
            t = ~t;
        ones = sz - ones;
    }

    bool all()
    {
        return ones == sz;
    }

    bool one()
    {
        return ones > 0;
    }

    int count()
    {
        return ones;
    }

    string toString()
    {
        string s;
        s.reserve(sz + 32);
        for (auto t : b)
        {
            for (int i = 0; i < 32; i++)
            {
                if (t & (1 << i))
                    s.push_back('1');
                else
                    s.push_back('0');
            }
        }
        return s.substr(0, sz);
    }
};

/**
 * Your Bitset object will be instantiated and called as such:
 * Bitset* obj = new Bitset(size);
 * obj->fix(idx);
 * obj->unfix(idx);
 * obj->flip();
 * bool param_4 = obj->all();
 * bool param_5 = obj->one();
 * int param_6 = obj->count();
 * string param_7 = obj->toString();
 */