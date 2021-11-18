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
    // 递归链表翻转
    ListNode* reverseK(ListNode* head, int k) {
        if (!head || !head->next) return head;
        int count = 0;
        auto p = head;
        while (true) {
            count++;
            if (!p->next || count == k) break;
            p = p->next;
        }
        if (count % 2 == 0) {
            auto pnext = p->next;
            p->next = nullptr;
            ListNode* h = head, *tail = nullptr;
            while (true) {
                auto t = h->next;
                h->next = tail;
                tail = h;
                if (!t) break;
                h = t;
            }
            head->next = reverseK(pnext, k+1);
            return h;
        } else {
            p->next = reverseK(p->next, k+1);
            return head;
        }
    }
public:
    ListNode* reverseEvenLengthGroups(ListNode* head) {
        return reverseK(head, 1); 
    }
};