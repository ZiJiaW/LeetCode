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
 * 深搜，记录当前子树内的是s还是d，以及记录路径即可
 */
class Solution {
    string res;
    string paths, pathd;
    string rvs(string s) {
        reverse(begin(s), end(s));
        return s;
    }
    int dfs(TreeNode* t, int s, int d, char dir) {
        if (!t) return 0;
        auto flagl = dfs(t->left, s, d, 'L');
        auto flagr = dfs(t->right, s, d, 'R');
        if (flagl == 0 && flagr == 0) {
            if (t->val == s) {
                paths.push_back('U');
                return 1;
            }
            if (t->val == d) {
                pathd.push_back(dir);
                return -1;
            }
            return 0;
        }
        if (flagl != 0 && flagr != 0) {
            res = paths + rvs(pathd);
            return 0;
        }
        int flag = flagl == 0 ? flagr : flagl;
        string &path = flag == 1 ? paths : pathd;
        if (t->val == s || t->val == d) {
            res = t->val == s ? rvs(path) : path;
            return 0;
        }
        if (flag == 1) {
            path.push_back('U');
            return 1;
        } else {
            path.push_back(dir);
            return -1;
        }
    }
public:
    string getDirections(TreeNode* root, int startValue, int destValue) {
        dfs(root, startValue, destValue, false);
        return res;
    }
};