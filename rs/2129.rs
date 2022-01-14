impl Solution {
    pub fn capitalize_title(title: String) -> String {
        let mut res = String::new();
        for mut s in title.to_ascii_lowercase().split(' ') {
            if s.len() <= 2 {
                res.push_str(s);
            } else {
                for (i, c) in s.char_indices() {
                    res.push(if i == 0 {
                        c.to_uppercase().next().unwrap()
                    } else {
                        c
                    });
                }
            }
            res.push(' ');
        }
        res.pop();
        res
    }
}
