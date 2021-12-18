/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution
{
    // 计数x所在结点的左子树大小和右子树大小，y选择最大的分支
public:
    bool btreeGameWinningMove(TreeNode *root, int n, int x)
    {
        int lsize = 0, rsize = 0;
        function<void(TreeNode *, int)> dfs = [&](TreeNode *t, int flag)
        {
            if (!t)
                return;
            if (flag < 0)
                lsize++;
            else if (flag > 0)
                rsize++;
            dfs(t->left, t->val == x ? -1 : flag);
            dfs(t->right, t->val == x ? 1 : flag);
        };
        dfs(root, 0);
        int psize = n - lsize - rsize - 1;
        return psize > n - psize || lsize > n - lsize || rsize > n - rsize;
    }
};