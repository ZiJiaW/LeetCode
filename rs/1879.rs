impl Solution {
    /*
    因为长度n不超过14，用bitmask k来表示使用nums2中的对应数字来xor nums1中前i个数字得到的最小值
    我们按bitmask的有效1的数量来遍历dp
    代码中，i为有效1的数量，也即nums1中前i个数字的匹配，寻找有效的bitmask k来遍历，
    找到一个k后，暴力尝试用k中的某位匹配nums1的第i位，剩下的位是dp已计算过的~
    */
    pub fn minimum_xor_sum(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
        let mut mem = HashMap::new();
        mem.insert(0, 0);
        let n = nums1.len();
        for i in 1..n + 1 {
            for k in 1..(1 << n) {
                if (k as i32).count_ones() == i as u32 {
                    let mut r = 1 << 30;
                    for j in 0..n {
                        if (k & (1 << j)) > 0 {
                            r = min(r, (nums1[i - 1] ^ nums2[j]) + mem[&(k ^ (1 << j))]);
                        }
                    }
                    mem.insert(k, r);
                }
            }
        }
        mem[&((1 << n) - 1)]
    }
}
