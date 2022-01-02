class Solution
{
    // bfs从supplies开始遍历，遍历到的结点的入度减一，入度减为0后才能往下走，最后输出所有入度为0的结点即可
public:
    vector<string> findAllRecipes(vector<string> &recipes, vector<vector<string>> &ingredients, vector<string> &supplies)
    {
        unordered_map<string, vector<int>> adj;
        unordered_map<string, int> indeg;
        vector<string> res;
        for (int i = 0; i < recipes.size(); i++)
        {
            indeg[recipes[i]] = ingredients[i].size();
            for (auto &&s : ingredients[i])
                adj[s].push_back(i);
        }
        queue<string> q;
        for (auto &&s : supplies)
            q.push(s);
        while (!q.empty())
        {
            string cur = move(q.front());
            q.pop();
            if (indeg[cur] > 0)
            {
                if (--indeg[cur] == 0)
                    res.push_back(cur);
            }
            if (adj.find(cur) != adj.end() && indeg[cur] == 0)
            {
                for (auto j : adj[cur])
                {
                    q.push(recipes[j]);
                }
            }
        }
        return res;
    }
};