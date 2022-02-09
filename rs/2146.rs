impl Solution {
    // 因为距离是第一优先级参数，因此广度优先搜索，并将记录到的四个维度加入优先队列，后取前k个点即可
    pub fn highest_ranked_k_items(
        mut grid: Vec<Vec<i32>>,
        pricing: Vec<i32>,
        start: Vec<i32>,
        k: i32,
    ) -> Vec<Vec<i32>> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, VecDeque};
        let (m, n) = (grid.len(), grid[0].len());
        let (sx, sy) = (start[0] as usize, start[1] as usize);
        let mut q = VecDeque::from(vec![(sx, sy, 0, grid[sx][sy])]);
        grid[sx][sy] = 0;
        let mut heap = BinaryHeap::new();
        let mut res = Vec::with_capacity(k as usize);
        while !q.is_empty() {
            let (x, y, dist, price) = q.pop_front().unwrap();
            if price >= pricing[0] && price <= pricing[1] {
                heap.push(Reverse((dist, price, x, y)));
            }
            if x + 1 < m && grid[x + 1][y] > 0 {
                q.push_back((x + 1, y, dist + 1, grid[x + 1][y]));
                grid[x + 1][y] = 0;
            }
            if x >= 1 && grid[x - 1][y] > 0 {
                q.push_back((x - 1, y, dist + 1, grid[x - 1][y]));
                grid[x - 1][y] = 0;
            }
            if y + 1 < n && grid[x][y + 1] > 0 {
                q.push_back((x, y + 1, dist + 1, grid[x][y + 1]));
                grid[x][y + 1] = 0;
            }
            if y >= 1 && grid[x][y - 1] > 0 {
                q.push_back((x, y - 1, dist + 1, grid[x][y - 1]));
                grid[x][y - 1] = 0;
            }
        }
        while !heap.is_empty() && res.len() < k as usize {
            let Reverse((_, _, x, y)) = heap.pop().unwrap();
            res.push(vec![x as i32, y as i32]);
        }
        res
    }
}
