impl Solution {
    // 先计算哪些地方可以敲章（正向前缀和记录右下角位置即可）
    // 然后再次计算哪些地方会被敲章覆盖（反向前缀和）
    pub fn possible_to_stamp(mut grid: Vec<Vec<i32>>, stamp_height: i32, stamp_width: i32) -> bool {
        use std::cmp::min;
        let (m, n) = (grid.len(), grid[0].len());
        let (h, w) = (stamp_height as usize, stamp_width as usize);
        let v = grid.clone();
        let mut stamp = vec![vec![0; n]; m];
        for i in 0..m {
            for j in 0..n {
                if i > 0 {
                    grid[i][j] += grid[i - 1][j];
                }
                if j > 0 {
                    grid[i][j] += grid[i][j - 1];
                }
                if i > 0 && j > 0 {
                    grid[i][j] -= grid[i - 1][j - 1];
                }
                if i >= h - 1 && j >= w - 1 {
                    let mut r = grid[i][j];
                    if i > h - 1 {
                        r -= grid[i - h][j];
                    }
                    if j > w - 1 {
                        r -= grid[i][j - w];
                    }
                    if i > h - 1 && j > w - 1 {
                        r += grid[i - h][j - w];
                    }
                    if r == 0 {
                        stamp[i][j] = 1; // can stamp at i, j
                    }
                }
            }
        }
        for i in (0..m).rev() {
            for j in (0..n).rev() {
                if i < m - 1 {
                    stamp[i][j] += stamp[i + 1][j];
                }
                if j < n - 1 {
                    stamp[i][j] += stamp[i][j + 1];
                }
                if i < m - 1 && j < n - 1 {
                    stamp[i][j] -= stamp[i + 1][j + 1];
                }
                if v[i][j] == 0 {
                    // try stamp
                    let x = min(m, i + h);
                    let y = min(n, j + w);
                    // from i,j to x-1, y-1
                    let mut r = stamp[i][j];
                    if x < m {
                        r -= stamp[x][j];
                    }
                    if y < n {
                        r -= stamp[i][y];
                    }
                    if x < m && y < n {
                        r += stamp[x][y];
                    }
                    if r == 0 {
                        return false;
                    }
                }
            }
        }
        true
    }
}
