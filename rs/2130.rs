// Definition for singly-linked list.
// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct ListNode {
//   pub val: i32,
//   pub next: Option<Box<ListNode>>
// }
//
// impl ListNode {
//   #[inline]
//   fn new(val: i32) -> Self {
//     ListNode {
//       next: None,
//       val
//     }
//   }
// }
impl Solution {
    pub fn pair_sum(mut head: Option<Box<ListNode>>) -> i32 {
        use std::cmp::max;
        let mut res = 0;
        let mut nums = Vec::with_capacity(10);
        while let Some(h) = head {
            nums.push(h.val);
            head = h.next;
        }
        let len = nums.len();
        for i in 0..len / 2 {
            res = max(res, nums[i] + nums[len - i - 1]);
        }
        res
    }
}
