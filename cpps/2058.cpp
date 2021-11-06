/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    // just traverse
    vector<int> nodesBetweenCriticalPoints(ListNode* head) {
        vector<int> res{-1, -1};
        ListNode *pre = nullptr;
        int first = -1, last = -1, pos = 0;
        for (auto p = head; p->next; pre = p, p = p->next, pos++) {
            if (!pre) continue;
            // pre p p->next
            if (pre->val < p->val && p->val > p->next->val ||
                pre->val > p->val && p->val < p->next->val) {
                // it is critical
                if (first == -1) first = pos;
                if (last != -1) {
                    res[0] = res[0] == -1 ? pos - last : min(res[0], pos - last);
                }
                last = pos;
            }
        }
        if (last != first && first != -1)
            res[1] = last - first;
        return res;
    }
};