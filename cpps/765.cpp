class Solution
{
    // 有关联的座位之间连一条边，则每次交换能够增加一个独立的连通分量，交换的目标是得到nCouple个连通分量
    // 因此只需要计算连通分量的数目即可，bfs/dfs/union-find均可
public:
    int minSwapsCouples(vector<int> &row)
    {
        int nCouple = row.size() / 2;
        vector<int> pos(row.size(), 0);
        for (int i = 0; i < nCouple; i++)
        {
            pos[row[2 * i]] = i;
            pos[row[2 * i + 1]] = i;
        }
        queue<int> q;
        vector<bool> v(nCouple, false);
        int res = 0;
        for (int i = 0; i < nCouple; i++)
        {
            if (!v[i])
            {
                res++;
                // bfs
                q.push(i);
                v[i] = true;
                while (!q.empty())
                {
                    int i = q.front();
                    q.pop();
                    int c1 = row[2 * i] % 2 == 0 ? pos[row[2 * i] + 1] : pos[row[2 * i] - 1];
                    int c2 = row[2 * i + 1] % 2 == 0 ? pos[row[2 * i + 1] + 1] : pos[row[2 * i + 1] - 1];
                    if (!v[c1])
                    {
                        v[c1] = true;
                        q.push(c1);
                    }
                    if (!v[c2])
                    {
                        v[c2] = true;
                        q.push(c2);
                    }
                }
            }
        }
        return nCouple - res;
    }
};