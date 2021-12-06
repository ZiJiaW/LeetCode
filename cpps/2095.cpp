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
    ListNode* deleteMiddle(ListNode* head) {
        int len = 0;
        for (auto p = head; p; p = p->next) len++;
        if (len == 1) {
            delete head;
            return nullptr;
        }
        int pos = len / 2;
        int cur = 0;
        for (auto p = head; p; p = p->next) {
            if (cur == pos - 1) {
                auto tmp = p->next;
                p->next = tmp->next;
                delete tmp;
                break;
            }
            cur++;
        }
        return head;
    }
};