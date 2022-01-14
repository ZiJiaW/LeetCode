impl Solution {
    // 用哈希表，没啥好说的
    pub fn longest_palindrome(words: Vec<String>) -> i32 {
        use std::collections::HashMap;
        let mut m = HashMap::new();
        for s in words {
            *m.entry(s).or_insert(0) += 1;
        }
        let mut res = 0;
        let mut sole = 0;
        let mut rev = vec![0; 2];
        for (k, v) in &m {
            let chs = k.as_bytes();
            if chs[0] == chs[1] {
                res += 4 * (v / 2);
                if v % 2 == 1 {
                    sole = 2;
                }
            } else if chs[0] < chs[1] {
                rev[0] = chs[1];
                rev[1] = chs[0];
                let newkey = String::from_utf8(rev.to_vec()).unwrap();
                if let Some(v2) = m.get(&newkey) {
                    res += 4 * v.min(v2);
                }
            }
        }
        res + sole
    }
}
