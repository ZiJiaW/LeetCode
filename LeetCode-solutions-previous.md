# LeetCode Solutions

[TOC]

代码附加极其简单的说明，个别极其简单的题目就不收录了。

## 10. Regular Expression Matching[DFS]

经典老题，需要考虑各种边界情况的DP，TopDown好写一点。

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        vector<vector<int>> dp(s.size()+1, vector<int>(p.size()+1, -1));
        dp[s.size()][p.size()] = 1;
        function<int(int, int)> dfs = [&](int i, int j) {
            if (dp[i][j] != -1) return dp[i][j];
            if (i == s.size()) {
                if (j < p.size() - 1 && p[j+1] == '*') return dp[i][j] = dfs(i, j+2);
                else return dp[i][j] = 0;
            }
            if (j == p.size()) return 0;
            if (j < p.size() - 1 && p[j+1] == '*') {
                if (s[i] == p[j] || p[j] == '.') {
                    if (dfs(i+1, j) > 0 || dfs(i, j+2) > 0) return dp[i][j] = 1;
                    else return dp[i][j] = 0;
                } else {
                    return dp[i][j] = dfs(i, j+2);
                }
            }
            if (s[i] == p[j] || p[j] == '.') return dp[i][j] = dfs(i+1, j+1);
            else return dp[i][j] = 0;
        };
        return dfs(0, 0) > 0;
    }
};
```

## 16. 3Sum Closest

首先排序，先确定第一个元素，而后在后续元素中寻找加和最接近剩余值的数对。这个子问题可以通过下面的想法得到：如果`nums[j]+nums[k]`比目标值小，则我们只能考虑将`j`后移使结果增大。反之亦然。从而子问题可在`O(k-j)`解决，算法复杂度为$O(n^2)$。

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int res = nums[0] + nums[1] + nums[2];
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size()-2; i++) {
            int j = i+1, k = nums.size()-1, t = target-nums[i];
            while (j < k) {
                int d = nums[j]+nums[k];
                if (abs(t-d) < abs(res-target)) res = nums[i]+d;
                if (d < t) j++;
                else k--;
            }
        }
        return res;
    }
};
```

##  32. Longest Valid Parentheses[Stack]

通过栈来操作，栈底永远是上次开始出现有效括号串的起始位置。在每次弹出一个左括号时，注意不计算当前这两个括号之间的距离，而是计算当前括号对长度以及之前连续的括号对长度之和，实际上就是弹出后的栈顶的一个未弹出的左括号到当前右括号的距离。这一步很tricky。如果不这么做的话，可以通过DP的思路，维护以括号i结尾的最长括号串，这通过栈来维护就很简单了，不过空间复杂度稍微大一点。

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> st;
        int res = 0;
        st.push(-1);
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ')') {
                if (st.size() > 1) {
                    st.pop();
                    res = max(res, i - st.top());
                } else st.top() = i;
            } else st.push(i);
        }
        return res;
    }
};
```

## 34. Find First and Last Position of Element in Sorted Array[Binary Search]

经典二分查找问题，相当于自己实现一下STL里的`lower_bound`和`upper_bound`。标准库的实现很巧妙，需要反复推敲。

```cpp
class Solution {
    using Iter = vector<int>::iterator;
    Iter find_left(Iter left, Iter right, int t) {
        int len = right - left;
        Iter it;
        while (len > 0) {
            it = left + len / 2;
            if (*it < t) {
                left = it + 1;
                len -= len/2 + 1;
            } else len = len/2;
        }
        return left;
    }
    Iter find_right(Iter left, Iter right, int t) {
        int len = right - left;
        Iter it;
        while (len > 0) {
            it = left + len/2;
            if (*it <= t) {
                left = it + 1;
                len -= len/2 + 1;
            } else len = len/2;
        }
        return left;
    }
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        auto l = find_left(nums.begin(), nums.end(), target);
        if (l == nums.end() || *l != target) return {-1, -1};
        auto r = find_right(l, nums.end(), target);
        return {static_cast<int>(l-nums.begin()), static_cast<int>(r-nums.begin()-1)};
    }
};
```

## 39. Combination Sum[DFS]

深搜。

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        function<void(int,vector<int>&,int)> dfs = [&](int i, vector<int>& tmp, int sum) {
            if (sum == target) {
                res.push_back(tmp);
                return;
            }
            if (i == candidates.size()) return;
            int sz = tmp.size();
            while (sum <= target) {
                dfs(i+1, tmp, sum);
                sum += candidates[i];
                tmp.push_back(candidates[i]);
            }
            tmp.resize(sz);
        };
        vector<int> tmp;
        dfs(0, tmp, 0);
        return res;
    }
};
```

## 40. Combination Sum II[DFS]

通过计数来防止重复。也可以排序后更精致的避免重复，不过懒得再写了。

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        unordered_map<int, int> m;
        for (auto i : candidates) m[i]++;
        function<void(vector<int>&,unordered_map<int, int>::iterator,int)> dfs = 
            [&](vector<int>& tmp, unordered_map<int, int>::iterator i, int sum) {
            if (sum == target) {
                res.push_back(tmp);
                return;
            }
            if (i == m.end()) return;
            int sz = tmp.size();
            for (int k = 0; k <= i->second && sum <= target; k++) {
                dfs(tmp, next(i), sum);
                sum += i->first;
                tmp.push_back(i->first);
            }
            tmp.resize(sz);
        };
        vector<int> tmp;
        dfs(tmp, m.begin(), 0);
        return res;
    }
};
```

## 42. Trapping Rain Water[Other]

保存左右侧的最高点即可计算储水量。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int lmax = 0, res = 0;
        vector<int> rmax(height.size(), 0);
        for (int i = height.size() - 2; i >= 0; i--) {
            rmax[i] = max(rmax[i+1], height[i+1]);
        }
        for (int i = 0; i < height.size(); i++) {
            if (lmax > height[i] && rmax[i] > height[i]) {
                res += min(lmax, rmax[i]) - height[i];
            }
            lmax = max(lmax, height[i]);
        }
        return res;
    }
};
```

进一步的问题在于如何把空间优化成$O(1)$，一个极其巧妙的方法：由于我们在计算储水量的时候只需要左右侧最高点中的较小值。而不一定需要计算好确切的最高点。例如`lmax`是左边`[0:i]`的确切的最高点，`rmax`是右边一部分`[j:n]`的最高点，此时如果`lmax<rmax`，我们可以确信当前`i`处需要加的是`lmax-height[i]`，反之我们计算`j`处的储水量。这样从两边开始向中间计算的话，我们不仅只需要一边扫描，而且也把空间复杂度优化掉了。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int lmax = 0, rmax = 0, res = 0;
        int i = 0, j = height.size() - 1;
        while (i <= j) {
            if (lmax < rmax) {
                if (lmax > height[i]) res += lmax - height[i];
                lmax = max(height[i++], lmax);
            } else {
                if (rmax > height[j]) res += rmax - height[j];
                rmax = max(height[j--], rmax);
            }
        }
        return res;
    }
};
```

## 85. Maximal Rectangle

DP，计算当前位置最大高度的矩形面积。

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;
        int maxArea = 0, m = matrix.size(), n = matrix[0].size();
        vector<int> R(n, -1);
        vector<int> L(n, n);
        vector<int> H(n, 0);
        for (int i = 0; i < m; ++i) {
            int left = n, right = -1;
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == '1') {
                    H[j] += 1;
                    left = min(j, left);
                    L[j] = H[j] == 1 || i == 0 ? left : max(L[j], left);
                } else {
                    H[j] = 0;
                    left = n;
                    L[j] = left;
                }
            }
            for (int j = 0; j < n; ++j) {
                if (matrix[i][n - j - 1] == '1') {
                    right = max(n - j - 1, right);
                    R[n - j - 1] = H[n - j - 1] == 1 || i == 0 ? right : min(R[n - j - 1], right);
                } else {
                    right = -1;
                    R[n - j - 1] = right;
                }
                maxArea = max(maxArea, (R[n - j - 1] - L[n - j - 1] + 1)*H[n - j - 1]);
            }
        }
        return maxArea;
    }
};
```



## 126. Word Ladder II

如果要寻找能否变换成目标串，则一次普通的BFS就可以解决问题。但是本题除此之外还需要输出所有能到达目标串的变换序列。因此需要对原来的BFS做些修改，我们在队列中存储变化的下标序列，同时为到达最短路，在向下拓展一层后，上层结点不可再次访问（而不是传统BFS刚加入队列就设为已访问）。与此同时，记录最短长度，一旦当前层数大于最短长度，可以直接返回。

```cpp
class Solution {
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        vector<vector<string>> res;
        if (find(begin(wordList), end(wordList), endWord) == wordList.end()) 
            return res;
        auto check = [](string& a, string& b) {
            int count = 0;
            for (int i = 0; i < a.size(); i++) count += a[i] != b[i];
            return count == 1;
        };
        wordList.push_back(beginWord);
        int n = wordList.size();
        vector<int> v(n, 0);
        vector<vector<int>> adj(n);
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                if (check(wordList[i], wordList[j])) {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }
        queue<vector<int>> q;
        q.push({n-1});
        int len = INT_MAX;
        while (!q.empty()) {
            auto t = q.front();
            q.pop();
            if (t.size() > len) break;
            int i = t.back();
            v[i] = 1;
            for (auto j : adj[i]) {
                if (!v[j]) {
                    if (wordList[j] == endWord) {
                        len = t.size();
                        t.push_back(j);
                        res.push_back(vector<string>());
                        for (auto k : t)
                            res.back().push_back(wordList[k]);
                    } else {
                        t.push_back(j);
                        q.push(t);
                        t.pop_back();
                    }
                }
            }
        }
        return res;
    }
};
```

## 133. Clone Graph

深搜遍历所有邻接结点，但注意需要用表来存储已经生成过得结点。

```cpp
class Solution {
    unordered_map<int, Node*> m;
public:
    Node* cloneGraph(Node* node) {
        if (!node) return nullptr;
        auto t = new Node(node->val);
        m[node->val] = t;
        for (Node* j : node->neighbors) {
            if (m.find(j->val) == m.end()) t->neighbors.push_back(cloneGraph(j));
            else t->neighbors.push_back(m[j->val]);
        }
        return t;
    }
};
```

## 137. Single Number II[State Machine]

我们设计一个计数器，计数每一位上出现1的次数，如果出现的次数模3余1，则最终结果上这一位就是1，这是很简单的想法。那么如何高效计数呢？最直观的想法是用一个32位的数组，但是这样的话计算量过大，我们可以设计一个有限状态自动机，在每一位上直接计算好模3的余数。其状态转移表如下：

| S1   | S0   | input | S1'  | S0'  |
| ---- | ---- | ----- | ---- | ---- |
| 0    | 0    | 0     | 0    | 0    |
| 0    | 0    | 1     | 0    | 1    |
| 0    | 1    | 0     | 0    | 1    |
| 0    | 1    | 1     | 1    | 0    |
| 1    | 0    | 0     | 1    | 0    |
| 1    | 0    | 1     | 0    | 0    |

想象对每一位上的输入都作为上面这个状态机的输入，最终达到的状态必然是模3的余数。上面这个状态机的转移函数也很好计算。由于有3个状态，分别代表模3余0,1和2。因此状态记录需要两位分别记录S1和S0，每个整数有32位，在每一位上的状态转移都是一致的，因此只要用两个32位整数记录即可。

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int s1 = 0, s0 = 0;
        for (auto i : nums) {
            int tmpS1 = (~s1&s0&i)|(s1&~s0&~i);
            s0 = (~s1&~s0&i)|(~s1&s0&~i);
            s1 = tmpS1;
        }
        return s0;
    }
};
```

## 141. Linked List Cycle

快慢指针找链表环：如果有环的话，两个指针都会进入环，那么快指针必然会在某一时刻追上慢指针。

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if (!head) return false;
        auto slow = head, fast = head;
        while (slow->next && fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
    }
};
```

## 155. Min Stack

每个位置额外存储当前的最小值即可。

```cpp
class MinStack {
    stack<pair<int,int>> s;
public:
    /** initialize your data structure here. */
    MinStack() {
        
    }
    
    void push(int x) {
        if (s.empty() || x < s.top().second) s.push({x,x});
        else s.push({x, s.top().second});
    }
    
    void pop() {
        s.pop();
    }
    
    int top() {
        return s.top().first;
    }
    
    int getMin() {
        return s.top().second;
    }
};
```

## 164. Maximum Gap

有趣，对n个数制作n+1个桶，每个桶存放一定区间内的数，这样至少有一个桶是空的，因此相邻桶间最大间隔一定大于桶内，我们只需计算相邻桶间的间隔即可。甚至桶都不用维护，只需要计算每个桶的最大元素和最小元素即可。

```cpp
class Solution {
public:
    int maximumGap(vector<int>& nums) {
        if (nums.size() < 2) return 0;
        int n = nums.size(), nMin = INT_MAX, nMax = INT_MIN;
        for (auto i : nums) {
            nMin = min(nMin, i);
            nMax = max(nMax, i);
        }
        int gap = (nMax - nMin)/(n+1) + 1;
        vector<int> mins(n+1, INT_MAX);
        vector<int> maxs(n+1, INT_MIN);
        int k;
        for (auto i : nums) {
            k = (i-nMin)/gap;
            mins[k] = min(mins[k], i);
            maxs[k] = max(maxs[k], i);
        }
        int lastMax = nMin, res = 0;
        for (int k = 0; k < n+1; k++) {
            if (mins[k] > maxs[k]) continue;
            res = max(res, mins[k] - lastMax);
            lastMax = maxs[k];
        }
        return res;
    }
};
```

## 174. Dungeon Game

假设$dp[i][j]$是骑士从$(i,j)$开始营救公主需要的最少生命值，则如果$dungeon[i][j]<0$，则生命值至少是$1-dungeon[i][j]$，再考虑$(i,j+1)$的情况，进入其中生命值至少是$dp[i][j+1]-dungeon[i][j]$，右侧情况同理，选择较小要求的生命值：
$$
dp[i][j]=max(1-dungeon[i][j],min(dp[i][j+1]-dungeon[i][j],dp[i+1][j]-dungeon[i][j]))
$$
对于$dungeon[i][j]\ge 0$的情况，我们必须以正生命值进入，因此至少是1，其余同理：
$$
dp[i][j]=max(1,min(dp[i][j+1]-dungeon[i][j],dp[i+1][j]-dungeon[i][j]))
$$

```cpp
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int m = dungeon.size(), n = dungeon[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for (int i = m-1; i >= 0; i--) {
            for (int j = n-1; j >= 0; j--) {
                dp[i][j] = dungeon[i][j] < 0 ? 1- dungeon[i][j] : 1;
                int down = i < m-1 ? dp[i+1][j] - dungeon[i][j] : INT_MAX;
                int right = j < n-1 ? dp[i][j+1] - dungeon[i][j] : INT_MAX;
                if (i < m-1 || j < n-1) dp[i][j] = max(dp[i][j], min(down, right));
            }
        }
        return dp[0][0];
    }
};
```

## 189. Rotate Array

三次reverse可以实现旋转。

```cpp
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        if (k == 0) return;
        reverse(begin(nums), begin(nums)+nums.size()-k);
        reverse(begin(nums)+nums.size()-k, end(nums));
        reverse(begin(nums), end(nums));
        return;
    }
};
```

## 201. Range Bitwise And

找到m和n第一个不相同的位，把后面清零即可。（因为进位会导致全部为零）

```cpp
class Solution {
public:
    int rangeBitwiseAnd(int m, int n) {
        int t = m^n, i = 0;
        while (t) {
            t >>= 1;
            i++;
        }
        return ((m&n)>>i)<<i;
    }
};
```

## 239. Sliding Window Maximum[Sliding window]

维持一个元素非增的双向队列，每次新添加的元素如果比队尾的元素大，则说明队尾元素不再可能成为最大值，将其删去。如此的话，队列头部元素即为最大值。考察内部的while循环，可以看出数组中每个元素最多被删除一次，从而两个循环加起来最多2n次，因此时间复杂度为$O(n)$。

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        deque<int> q;
        vector<int> res;
        for (int i = 0; i < n; ++i)
        {
            if (!q.empty() && q.front() == i - k) q.pop_front();
            while (!q.empty() && nums[q.back()] <= nums[i]) q.pop_back();
            q.push_back(i);
            if (i >= k - 1) res.push_back(nums[q.front()]);
        }
        return res;
    }
};
```

## 260. Single Number III

首先我们对所有数做异或，则得到了要找的两个数的异或结果res。res中至少有1个位为1，否则两个数会相同。找到最低位的1，说明a和b在这个位上不同，而所有其他数在这个位上是有配对的，因此我们只需要把所有在这个位上同为1或0的数异或，就一定能找到a和b之中的一个，最后通过异或把另一个找出来即可。

```CPP
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        int res = 0;
        for (auto i : nums)
            res ^= i;
        int mask = res & ~(res - 1);
        int res0 = 0;
        for (auto i : nums) {
            if (mask & i) res0 ^= i;
        }
        return {res0, res0^res};
    }
};
```

## 295. Find Median from Data Stream[Heap]

维护最大堆和最小堆，并维持两者大小差不超过1，则堆顶元素动态确定中位数。

```cpp
class MedianFinder {
    priority_queue<int> maxheap;
    priority_queue<int, vector<int>, greater<int>> minheap;
public:
    MedianFinder() {}
    
    void addNum(int num) {
        if (maxheap.empty()) maxheap.push(num);
        else if (num <= maxheap.top()) maxheap.push(num);
        else minheap.push(num);
        if (maxheap.size() > minheap.size() + 1) {
            minheap.push(maxheap.top());
            maxheap.pop();
        } else if (minheap.size() > maxheap.size() + 1) {
            maxheap.push(minheap.top());
            minheap.pop();
        }
    }
    
    double findMedian() {
        if (maxheap.size() == minheap.size()) 
            return (static_cast<double>(maxheap.top()) + static_cast<double>(minheap.top()))/2.0;
        else if (maxheap.size() > minheap.size()) return maxheap.top();
        else return minheap.top();
    }
};
```

## 312. Burst Balloons[DP]

设$dp[i][j]$为`i`至`j`号气球打破的最大得分，`l`和`r`是区间的左右界，我们要保证`l=nums[i-1]`和`r=nums[j+1]`，不能在中间左右界出现变化，因此考虑期间最晚打破的气球`k`，其必然成为区间`[i,k-1]`和`[k+1,j]`上气球的右界和左界，且处理子问题的时候这个界不改变。因此有：
$$
dp[i][j]=\max_{k=i}^{j}\{dp[i][k-1]+dp[k+1][j]+nums[k]*nums[i-1]*nums[j+1]\}
$$

```cpp
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        vector<vector<int>> dp(n+1, vector<int>(n, 0));
        for (int i = n-1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                int left = i == 0 ? 1 : nums[i-1];
                int right = j == n - 1 ? 1 : nums[j+1];
                dp[i][j] = left*nums[i]*right + dp[i+1][j];
                for (int k = i + 1; k <= j; k++) {
                    dp[i][j] = max(left*right*nums[k]+dp[i][k-1]+dp[k+1][j], dp[i][j]);
                }
            }
        }
        return dp[0][n-1];
    }
};
```

## 315. Count Smaller

分治排序的另一种应用，只不过这次我们从大到小排，记录对应index的右侧较小值的数目。

```cpp
class Solution {
public:
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> res(nums.size(), 0);
        vector<int> idx(nums.size(), 0);
        for (int i = 0; i < nums.size(); i++) idx[i] = i;
        vector<int> tmp;
        function<void(int,int)> mergeSort = [&](int i, int j) {
            if (j-i < 1) return;
            int mid = i + (j-i)/2;
            mergeSort(i, mid);
            mergeSort(mid+1, j);
            tmp.clear();
            int p = i, q = mid+1;
            while (p <= mid && q <= j) {
                if (nums[idx[p]] > nums[idx[q]]) {
                    res[idx[p]] += j-q+1;
                    tmp.push_back(idx[p++]);
                } else {
                    tmp.push_back(idx[q++]);
                }
            }
            while (q <= j) tmp.push_back(idx[q++]);
            while (p <= mid) tmp.push_back(idx[p++]);
            copy(tmp.begin(), tmp.end(), idx.begin()+i);
        };
        mergeSort(0, nums.size() - 1);
        return res;
    }
};
```

## 335. Self Crossing[Other]

由于是逆时针，每次移动都可以对称的看做是第一次向上开始，在纸上画几次就可以找到规律了，一共两种发生交叉的情况。需要注意的是需要在后面补几个0，以避免某些边缘的情况（比如恰好在一个点处相遇）。

```cpp
class Solution {
public:
    bool isSelfCrossing(vector<int>& x) {
        for (int i = 0; i < 6; i++) x.push_back(0);
        for (int i = 0; i < x.size(); i++) {
            if (i + 3 < x.size()) {
                if (x[i+3] >= x[i+1] && x[i+2] <= x[i] && x[i+1] > 0)
                    return true;
            }
            if (i + 5 < x.size()) {
                if (x[i+3] >= x[i+1] && x[i+1] > 0 && x[i+5] >= x[i+3] - x[i+1] 
                    && x[i+4] >= x[i+2] - x[i] && x[i+2] >= x[i+4])
                    return true;
            }
        }
        return false;
    }
};
```

## 410. Split Array Largest Sum

二分查找，寻找能够bound住m个子序列和的最小上界。确定上界后，贪心地尝试分划子序列即可。

```cpp
class Solution {
public:
    int splitArray(vector<int>& nums, int m) {
        int64_t hi = 0, lo;
        int Max = 0;
        for (auto x : nums) {
            hi += x;
            Max = max(Max, x);
        }
        lo = max(hi/m, (int64_t)Max);
        while (hi - lo > 0) {
            int64_t mid = lo + (hi-lo)/2;
            int64_t sum = 0;
            int count = 1;
            for (auto x : nums) {
                sum += x;
                if (sum > mid) {
                    sum = x;
                    if (++count > m)
                        break;
                }
            }
            if (count > m) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
};
```

## 427. Construct Quad Tree[Divide and Conquer]

分治，需要注意的是叶子结点可以不用单独生成，他们都是一样的。

```cpp
/*
// Definition for a QuadTree node.
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;

    Node() {}

    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};
*/
class Solution {
public:
    Node* construct(vector<vector<int>>& grid) {
        return divide(0, 0, grid.size(), grid);
    }
    
    Node* tleaf = new Node(true, true, nullptr, nullptr, nullptr, nullptr);
    Node* fleaf = new Node(false, true, nullptr, nullptr, nullptr, nullptr);
    
    Node* divide(int x, int y, int sideLen, vector<vector<int>>& grid) {
        if (sideLen == 1) {
            return grid[x][y] == 1 ? tleaf : fleaf;
        }
        int newlen = sideLen / 2;
        auto topleft = divide(x, y, newlen, grid);
        auto topright = divide(x, y+newlen, newlen, grid);
        auto bottomleft = divide(x+newlen, y, newlen, grid);
        auto bottomright = divide(x+newlen, y+newlen, newlen, grid);
        if (topleft->isLeaf && topright->isLeaf && bottomleft->isLeaf && bottomright->isLeaf
          &&topleft->val == topright->val && topleft->val == bottomleft->val && topleft->val == bottomright->val) {
            return topleft->val == true ? tleaf : fleaf;
        }
        return new Node(false, false, topleft, topright, bottomleft, bottomright);
    }
};
```

## 451. Sort Characters By Frequency[Sort]

排序。

```cpp
class Solution {
public:
    string frequencySort(string s) {
        unordered_map<char, int> m;
        string x;
        for (char c : s) {
            m[c]++;
            if (m[c] == 1) x += c;
        }
        auto cmp = [&](auto a, auto b) {
            return m[a] > m[b];
        };
        sort(x.begin(), x.end(), cmp);
        string r;
        for (auto c : x) {
            while (m[c] > 0) {
                r += c;
                m[c]--;
            }
        }
        return r;
    }
};
```

## 452. Minimum Number of Arrows to Burst Balloons

先区间排序，因为每一个气球都必须打到，维护可一次性打破的气球集合的最小右边界，当遇到新的左边界大于最小右边界时，把目前累积的气球全部打破，而后从新的气球重新开始。

```cpp
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if (points.size() <= 1) return points.size();
        sort(points.begin(), points.end(),
        [](const vector<int> &vec1, const vector<int> &vec2) {
            return vec1[0] < vec2[0];
        });
        int res = 1;
        int right = points[0][1];
        for (int i = 1; i < points.size(); i++) {
            if (points[i][0] > right) {
                res++;
                right = points[i][1];
            } else if (points[i][1] < right) {
                right = points[i][1];
            }
        }
        return res;
    }
};
```

## 456. 132 Pattern

从左向右扫描，保存前i-1元素的最大值和最小值，而后从i往前寻找第一个比i大的元素j，判断j前的元素最小值。

```cpp
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        int n = nums.size();
        if (n <= 2) return false;
        vector<int> mins(n, 0);
        int maxx = max(nums[0], nums[1]);
        mins[0] = nums[0];
        mins[1] = min(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            if (maxx > nums[i] && mins[i-1] < nums[i]) {
                int j = 0;
                for (j = i-1; j > 0; j--) {
                    if (nums[j] > nums[i]) {
                        if (mins[j-1] < nums[i]) return true;
                        break;
                    }
                }
            }
            maxx = max(maxx, nums[i]);
            mins[i] = min(mins[i-1], nums[i]);
        }
        return false;
    }
};
```

单调栈可以将复杂度降为O(n)，思路是先计算前缀最小值，然后从后往前维护递减栈。

```cpp
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        if (nums.size() < 3) return false;
        stack<int> s;
        vector<int> mins(nums.size(), nums[0]);
        for (int i = 1; i < nums.size(); i++) {
            mins[i] = min(nums[i], mins[i-1]);
        }
        for (int i = nums.size() - 1; i >= 0; i--) {
            while (!s.empty() && nums[i] > s.top()) {
                if (mins[i] < s.top()) return true;
                s.pop();
            }
            s.push(nums[i]);
        }
        return false;
    }
};
```

## 457. Circular Array Loop

深搜找环（后向边），去掉不同“方向”的环和自己到自己的边。

```cpp
class Solution {
public:
    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        enum {WHITE = 0, GREY, BLACK};
        vector<int> visited(n, WHITE);
        bool res = false;
        function<void(int, bool)> dfs = [&](int i, bool forward) {
            if ((forward && nums[i] < 0) || (!forward && nums[i] > 0)) return;
            if (visited[i] == GREY) {
                res = true;
            }
            if (visited[i] != WHITE) return;
            visited[i] = GREY;
            int next = (i+nums[i]) % n;
            next = next < 0 ? next + n : next;
            if (next != i) dfs(next, forward);
            visited[i] = BLACK;
        };
        for (int i = 0; i < n; i++) {
            if (visited[i] == WHITE) dfs(i, nums[i] > 0);
        }
        return res;
    }
};
```

## 459 Repeated Substring Pattern

判断字符串是否由周期子串组成。使用KMP算法生成前缀数组。最后的判断比较tricky，就是说明某个字符串的前缀等于后缀，则如果除去前缀剩余的部分长度整除字符串的长度，则字符串是重复的。设$S=AB$，其中$A$是`next[n-1]`作为尾部的前缀，同时也是$S$的后缀，则导出$B$也是$A$的后缀。此时将$B$从尾部截去，剩余部分是$S=A$，$A$的剩余部分也是$A$的前缀加后缀，满足上面的条件，也就是说$B$还是$S$的后缀，重复上述步骤可以说明子串由$B$重复。

```cpp
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        int j = 0, i = 1;
        vector<int> next(n, 0);
        while (i < n) {
            if (s[i] == s[j]) {
                next[i++] = ++j;
            } else {
                if (j > 0) j = next[j-1];
                else next[i++] = 0;
            }
        }
        return next[n-1]!=0 && n%(n-next[n-1])==0;
    }
};
```

## 460. LFU Cache

经典数据结构题，二维链表分别维护使用次数的序和使用时间的序，用哈希表存储内部元素在链表中的指针。

```cpp
class LFUCache {
    int _capacity;
    int _size;
    struct node {
        int count;
        list<pair<int, int>> vals;
        node(int n): count(n) {}
    };
    list<node> _caches;
    using niter = list<node>::iterator;
    using piter = list<pair<int, int>>::iterator;
    unordered_map<int, pair<niter, piter>> _map;
public:
    LFUCache(int capacity): _capacity(capacity), _size(0) {}
    
    int get(int key) {
        if (_map.find(key) == _map.end()) return -1;
        auto itx = _map[key].first;
        auto ity = _map[key].second;
        int val = ity->second;
        int old_count = itx->count;
        itx->vals.erase(ity);
        if (itx->vals.empty()) {
            itx = _caches.erase(itx);
        } else {
            itx++;
        }
        if (itx == _caches.end() || itx->count != old_count + 1) {
            itx = _caches.insert(itx, node(old_count + 1));
        }
        ity = itx->vals.insert(itx->vals.end(), make_pair(key, val));
        _map[key] = make_pair(itx, ity);
        return val;
    }
    
    void put(int key, int value) {
        if (_capacity == 0) return;
        if (_map.find(key) == _map.end()) {
            if (_size == _capacity) {
                auto itx = _caches.begin();
                _map.erase(itx->vals.begin()->first);
                itx->vals.pop_front();
                if (itx->vals.empty()) {
                    _caches.erase(itx);
                }
                _size--;
            }
            _size++;
            auto itx = _caches.begin();
            if (itx->count != 1) {
                itx = _caches.insert(itx, node(1));
            }
            auto ity = itx->vals.insert(itx->vals.end(), make_pair(key, value));
            _map[key] = make_pair(itx, ity);
        } else {
            auto ity = _map[key].second;
            ity->second = value;
            get(key);
        }
    }
};
```

## 462. Minimum Moves to Equal Array Elements II

经过简单的计算，可以发现找中位数作为最终的收敛值即可。

```cpp
class Solution {
public:
    int minMoves2(vector<int>& nums) {
        int n = nums.size();
        nth_element(nums.begin(), nums.begin() + n/2, nums.end());
        int t = nums[n/2];
        int res = 0;
        for (auto num : nums) {
            res += abs(t - num);
        }
        return res;
    }
};
```

## 467. Unique Substrings in Wraparound String

记录各字母开头的合法子串的最大长度，而后相加，记录过程len的计算采用DP的思路。

```cpp
class Solution {
public:
    int findSubstringInWraproundString(string p) {
        int res = 0;
        vector<int> r(26, 0);
        int len = 0;
        for (int i = p.size() - 1; i >= 0; i--) {
            if (i == p.size() - 1) {
                len = 1;
                r[p[i] - 'a'] = 1; 
            } else {
                if (p[i] == p[i+1] - 1 || p[i] == p[i+1] + 25) {
                    len++;
                } else {
                    len = 1;
                }
                r[p[i] - 'a'] = max(len, r[p[i] - 'a']);
            }
        }
        for (auto n : r) res += n;
        return res;
    }
};
```

## 470. Implement Rand10() Using Rand7()

建一张7*7二维表，随机指定其中一个点，其中前40个点概率一样，可以作为最终的随机结果。

```cpp
// The rand7() API is already defined for you.
// int rand7();
// @return a random integer in the range 1 to 7

class Solution {
public:
    int rand10() {
        int row, col, idx;
        do {
            row = rand7();
            col = rand7();
            idx = col + (row - 1) * 7;
        } while (idx > 40);
        return 1 + (idx - 1) % 10;
    }
};
```

## 472. Concatenated Words

建前缀树然后在其上深搜；更简单做法：哈希表存储所有字符串，暴力深搜查询子串是否可分。

```cpp
class Solution {
    struct TrieNode {
        bool isLeaf;
        vector<TrieNode*> childs;
        TrieNode(): isLeaf(false), childs(26, nullptr) {}
    };
public:
    void add(string& s, TrieNode* t) {
        for (auto c : s) {
            if (!t->childs[c-'a']) t->childs[c-'a'] = new TrieNode();
            t = t->childs[c-'a'];
        }
        t->isLeaf = true;
    }
    vector<int> searchPrefix(int start, string& s, TrieNode* t) {
        vector<int> nexts;
        while (t->childs[s[start]-'a']) {
            t = t->childs[s[start++]-'a'];
            if (t->isLeaf) nexts.push_back(start);
            if (start == s.size()) break;
        }
        return nexts;
    }
    bool dfs(int start, string& s, TrieNode* t) {
        if (s.size() == 0) return false;
        if (start == s.size()) return true;
        auto nexts = searchPrefix(start, s, t);
        for (auto next : nexts) {
            if (dfs(next, s, t)) return true;
        }
        return false;
    }
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        if (words.size() < 3) return {};
        sort(words.begin(), words.end(), [](string& a, string& b) {
            return a.size() < b.size();
        });
        TrieNode* trie = new TrieNode();
        vector<string> res;
        for (auto word : words) {
            if (dfs(0, word, trie)) res.push_back(word);
            else add(word, trie);
        }
        return res;
    }
};
```

## 473. Matchsticks to Square

本题用深搜，速度的关键在于剪枝，首先排序（先用长的火柴），其次去掉重复长度（因为四条边的等价性），最后预判当前长度是否可行（剩余最短火柴和其组合后，边长是否超过sum）。

```cpp
class Solution {
public:
    bool makesquare(vector<int>& nums) {
        if (nums.size() < 4) return false;
        int sum = 0;
        for (auto num : nums) {
            sum += num;
        }
        if (sum % 4 != 0) return false;
        sum /= 4;
        sort(nums.begin(), nums.end(), [](int a, int b) {
            return a > b;
        });
        int nmin = nums[nums.size() - 1];
        vector<int> r(4, 0);
        function<bool(int)> dfs = [&](int i) {
            if (i == nums.size()) return true;
            unordered_set<int> rep;
            for (int k = 0; k < 4; k++) {
                if (rep.find(r[k]) == rep.end() && r[k] + nums[i] <= sum) {
                    rep.insert(r[k]);
                    if (r[k] + nums[i] < sum && r[k] + nums[i] + nmin > sum) continue;
                    r[k] += nums[i];
                    if (dfs(i+1)) return true;
                    r[k] -= nums[i];
                }
            }
            return false;
        };
        return dfs(0);
    }
};
```

## 474. Ones and Zeroes

0-1背包的变种，依然是DP，$O(Lmn)$。值得一提的是可以把第一维优化，将空间复杂度降至$O(mn)$。

```cpp
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        int len = strs.size();
        if (len == 0) return 0;
        vector<vector<vector<int>>> dp(len+1, vector<vector<int>>(m+1, vector<int>(n+1, 0)));
        for (int i = 1; i <= len; i++) {
            int count0 = count(strs[i-1].begin(), strs[i-1].end(), '0');
            int count1 = strs[i-1].size() - count0;
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n; k++) {
                    if (j >= count0 && k >= count1) {
                        dp[i][j][k] = max(dp[i-1][j][k], dp[i-1][j-count0][k-count1] + 1);
                    } else {
                        dp[i][j][k] = dp[i-1][j][k];
                    }
                }
            }
        }
        return dp[len][m][n];
    }
};
```

由于每次更新会用到二维表左上方的旧值，因此缩减维度后需要从右下方开始更新，防止使用新值：

```cpp
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        if (strs.empty()) return 0;
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for (auto str : strs) {
            int count0 = count(str.begin(), str.end(), '0');
            int count1 = str.size() - count0;
            for (int j = m; j >= 0; j--) {
                for (int k = n; k >= 0; k--) {
                    if (j >= count0 && k >= count1) {
                        dp[j][k] = max(dp[j][k], dp[j-count0][k-count1] + 1);
                    }
                }
            }
        }
        return dp[m][n];
    }
};
```

## 475. Heaters

对加热器排序，寻找距离房子最近的来计算距离，返回最大的，搜索过程使用二分查找。

```cpp
class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        if (houses.size() == 0) return 0;
        sort(heaters.begin(), heaters.end());
        int radius = 0;
        for (auto house : houses) {
            auto i = lower_bound(heaters.begin(), heaters.end(), house);
            if (i == heaters.begin()) radius = max(radius, *i - house);
            else if (i == heaters.end()) radius = max(radius, house - *heaters.rbegin());
            else radius = max(radius, min(*i - house, house - *(i-1)));
        }
        return radius;
    }
};
```

## 476. Number Complement

从高位开始尝试翻转。更简洁做法：将~0左移直到与num与为零，就是前导零的对应数量的前导一。

```cpp
class Solution {
public:
    int findComplement(int num) {
        bool zero = true;
        for (int i = 31; i >= 0; i--) {
            if (num & (1 << i)) {
                zero = false;
                num &= ~(1 << i);
            } else if (!zero) {
                num |= 1 << i;
            }
        }
        return num;
    }
};
```

```cpp
class Solution {
public:
    int findComplement(int num) {
        unsigned int mask = ~0;
        while (num & mask) mask <<= 1;
        return ~mask & ~num;
    }
};
```

## 477. Total Hamming Distance

计算所有位上出现1的次数，将0的次数和1的次数相乘就是该位产生的所有两两间的Hamming距离。

```cpp
class Solution {
public:
    int totalHammingDistance(vector<int>& nums) {
        if (nums.size() <= 1) return 0;
        int res = 0;
        int count1 = 0;
        for (int i = 0; i < 32; i++) {
            count1 = 0;
            for (auto num : nums) {
                count1 += (num >> i) & 1; 
            }
            res += count1 * (nums.size() - count1);
        }
        return res;
    }
};
```

## 478. Generate Random Point in a Circle

先均匀随机生成正方形内的点，再剔除不再圆内的点，从而保证每个点生成概率相同。

```cpp

class Solution {
    double _x;
    double _y;
    double _r;
    default_random_engine gen;
    uniform_real_distribution<> randx;
    uniform_real_distribution<> randy;
public:
    Solution(double radius, double x_center, double y_center):
    _x(x_center),
    _y(y_center),
    _r(radius),
    randx(_x - _r, _x + _r),
    randy(_y - _r, _y + _r) {}
    
    vector<double> randPoint() {
        while (true) {
            double x = randx(gen);
            double y = randy(gen);
            if ((x-_x)*(x-_x)+(y-_y)*(y-_y) <= _r*_r) return {x,y};
        }
    }
};
```

## 479. Largest Palindrome Product

暴力求解，拼接回文数，看是否是两个n位数的乘积。这个解法基于这样的观察：两个n位数乘积的最大回文数是2n位的。

```cpp
class Solution {
public:
    int largestPalindrome(int n) {
        if (n == 1) return 9;
        int upper = pow(10, n) - 1;
        int lower = pow(10, n-1);
        for (int i = upper; i >= lower; i--) {
            string s = to_string(i);
            string r = s;
            reverse(r.begin(), r.end());
            long palin = stol(s+r);
            long sq = sqrt(palin);
            for (long k = upper; k >= sq; k--) {
                if (palin % k == 0 && palin/k < upper) 
                    return palin%1337;
            }
        }
        return -1;
    }
};
```

## 480. Sliding Window Median

借鉴295题的思路，由于cpp的堆没有删除操作，因此用红黑树来模拟，从而添加`removeNum`方法。也可以自己手撸一个Heap。

```cpp
class Solution {
    class MedianFinder {
        multiset<int> minheap;
        multiset<int, greater<int>> maxheap;
        void check() {
            if (maxheap.size() > minheap.size() + 1) {
                minheap.insert(*maxheap.begin());
                maxheap.erase(maxheap.begin());
            } else if (minheap.size() > maxheap.size() + 1) {
                maxheap.insert(*minheap.begin());
                minheap.erase(minheap.begin());
            }
        }
    public:
        /** initialize your data structure here. */
        MedianFinder() {}

        void addNum(int num) {
            if (maxheap.empty() || num <= *maxheap.begin()) maxheap.insert(num);
            else minheap.insert(num);
            check();
        }
        
        void removeNum(int num) {
            auto it = maxheap.find(num);
            if (it != maxheap.end()) {
                maxheap.erase(it);
            }
            else if ((it = minheap.find(num)) != minheap.end()) {
                minheap.erase(it);
            }
            check();
        }

        double findMedian() {
            if (maxheap.size() == minheap.size()) 
                return (static_cast<double>(*maxheap.begin()) + static_cast<double>(*minheap.begin()))/2.0;
            else if (maxheap.size() > minheap.size()) return *maxheap.begin();
            else return *minheap.begin();
        }
    };
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        MedianFinder mf;
        vector<double> res;
        for (int i = 0; i < nums.size(); i++) {
            if (i >= k) {
                mf.removeNum(nums[i-k]);
            }
            mf.addNum(nums[i]);
            if (i >= k - 1) {
                res.push_back(mf.findMedian());
            }
        }
        return res;
    }
};
```

## 481. Magical String

稍微演算一下就能找到生成这种字符串的办法了。这里我用队列减少了空间复杂度，不用存整个字符串了。

```cpp
class Solution {
public:
    int magicalString(int n) {
        if (n == 0) return 0;
        else if (n <= 3) return 1;
        int res = 1, i = 3;
        queue<int> q({2});
        while (i < n) {
            int j = q.front();
            if (q.back() == 1) {
                i += j;
                while ((j--) > 0) {
                    q.push(2);
                }
            } else {
                i += j;
                res += j;
                while ((j--) > 0) {
                    q.push(1);
                }
                if (i > n) res--;
            }
            q.pop();
        }
        return res;
    }
};
```

## 483. Smallest Good Base

演算出简单的上下界后，按位数m从大到小，通过二分查找的方式确定k进制。如果位数不大于2，就返回n-1。

```cpp
class Solution {
public:
    string smallestGoodBase(string n) {
        long long int num = stol(n);
        for (int m = log(num+1) / log(2); m > 2; m--) {
            long long l = 2, r = pow(num, 1.0 / (m - 1)) + 1;
            while (l < r) {
                long long k = l + (r - l) / 2;
                long long sum = 0, tmp = 1;
                for (int j = 0; j < m; j++) {
                    sum += tmp;
                    if (sum > num) {
                        r = k;
                        break;
                    }
                    if (j == m - 1) {
                        if (sum < num) {
                            l = k + 1;
                        } else {
                            return to_string(k);
                        }
                        break;
                    }
                    tmp *= k;
                }
            }
        }
        return to_string(num - 1);
    }
};
```

## 485. Max Consecutive Ones

滑动计数。

```cpp
class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int res = 0, count = 0;
        for (auto num : nums) {
            if (num == 0) {
                res = max(res, count);
                count = 0;
            } else {
                count++;
            }
        }
        res = max(res, count);
        return res;
    }
};
```

## 486. Predict the Winner

设`dp[i][j]`为玩家从区间`[i,j]`开始选，能够比另一个玩家多出来的最大分数。显然，当该玩家选完后，下一个玩家是对方，因此取反，并加上新选择的分数：
$$
dp[i][j]=\max\{nums[i]-dp[i+1][j],nums[j]-dp[i][j-1]\}
$$

```cpp
class Solution {
public:
    bool PredictTheWinner(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int k = 0; k < n; k++) {
            for (int i = 0; i + k < n; i++) {
                dp[i][i+k] = k == 0 ? nums[i] :
                max(nums[i]-dp[i+1][i+k], nums[i+k]-dp[i][i+k-1]);
            }
        }
        return dp[0][n-1] >= 0;
    }
};
```

## 488. Zuma Game

写起来很烦的一题，主要是深搜，插入后的“消消乐”用栈实现，优化有四点：

* 对hand排序后用哈希表记录重复状态。
* 记录全局最优值，剪掉比当前最优差的。
* 不重复在相同的字母的前后插入。
* 在不同的字母处插入时不用消减。

```cpp
class Solution {
public:
    int findMinStep(string board, string hand) {
        int res = INT_MAX;
        unordered_map<char, int> count;
        for (auto c : board) count[c]++;
        for (int i = 0; i < hand.size(); i++) {
            if (count.find(hand[i]) == count.end()) {
                hand.erase(i, 1);
                i--;
            } else {
                count[hand[i]]++;
            }
        }
        for (auto m : count) if (m.second < 3) return -1;
        sort(hand.begin(), hand.end());
        unordered_set<string> mem;
        dfs(board, hand, 0, res, mem);
        return res == INT_MAX ? -1 : res;
    }
    
    string eliminate(string& s) {
        vector<pair<char, int>> stk;
        for (auto c : s) {
            if (stk.empty()) {
                stk.push_back({c, 1});
            } else if (c == stk.back().first) {
                stk.rbegin()->second++;
            } else {
                if (stk.back().second >= 3) {
                    stk.pop_back();
                }
                if (!stk.empty() && c == stk.back().first) stk.rbegin()->second++;
                else stk.push_back({c, 1});
            } 
        }
        if (stk.back().second >= 3) {
            stk.pop_back();
        }
        string t;
        for (auto p : stk) {
            while (p.second > 0) {
                t.push_back(p.first);
                p.second--;
            }
        }
        return t;
    }
    
    void dfs(string s, string hand, int depth, int& gdepth, unordered_set<string>& mem) {
        if (s.empty()) {
            gdepth = min(gdepth, depth);
            return;
        }
        if (depth >= gdepth - 1) return;
        if (mem.find(s + "|" + hand) != mem.end()) return;
        mem.insert(s + "|" + hand);
        for (int k = 0; k < hand.size(); k++) {
            for (int i = 0; i < s.size(); i++) {
                string ns = s;
                ns.insert(i, 1, hand[k]);
                if (s[i] == hand[k]) {
                    i++;
                    if (i+1 < s.size() && s[i+1] == hand[k]) i++;
                    ns = eliminate(ns);
                }
                dfs(ns, hand.substr(0, k) + hand.substr(k+1), depth + 1, gdepth, mem);
            }
        }
    }
};
```

## 491. Increasing Subsequences

深搜，因为是求所有的解，感觉没啥好优化的。

```cpp
class Solution {
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        set<vector<int>> res;
        function<void(int, vector<int>&)> dfs = [&](int i, vector<int>& tmp) {
            if (tmp.size() >= 2) res.insert(tmp);
            for (int j = i; j < nums.size(); j++) {
                if (tmp.empty() || tmp.back() <= nums[j]) {
                    tmp.push_back(nums[j]);
                    dfs(j+1, tmp);
                    tmp.pop_back();
                }
            }
        };
        vector<int> tmp;
        dfs(0, tmp);
        return vector<vector<int>>(res.begin(), res.end());
    }
};
```

## 493. Reverse Pairs

使用分治排序，在merge步骤中添加逆序对的计数，复杂度为$\Theta(n\log n)$。使用一个全局的tmp数组可以加速。

```cpp
class Solution {
    vector<int> tmp;
public:
    int reversePairs(vector<int>& nums) {
        tmp.resize(nums.size());
        function<int(int, int)> count = [&](int left, int right) {
            if (right - left <= 1) return 0;
            int mid = left + (right - left) / 2;
            int res = count(left, mid) + count(mid, right);
            int k = 0, i = left, j = mid;
            while (i < mid && j < right) {
                if (nums[i] > (long)2 * nums[j]) {
                    res += mid - i;
                    j++;
                } else i++;
            }
            i = left; j = mid;
            while (k < right - left) {
                if (i < mid && j < right) {
                    if (nums[i] >= nums[j]) tmp[k++] = nums[j++];
                    else tmp[k++] = nums[i++];
                } else if (i < mid) tmp[k++] = nums[i++];
                else tmp[k++] = nums[j++];
            }
            copy(tmp.begin(), tmp.begin() + right - left, nums.begin() + left);
            return res;
        };
        return count(0, nums.size());
    }
};
```

## 494. Target Sum

本题有三种解法，第一种我尝试深搜，通过排序+剪枝达到184ms。第二种，显然这题可以DP，类似背包问题，假设数组长度为N，数组元素和为M，复杂度为$O(NM)$，写出来是36ms。第三种是更巧妙的DP，假设数组被分成两部分，一部分取正，一部分取负，其部分和为$A$和$B$，则最终有$A-B=S$，显然因为$M=A+B$，因此$A=(M+S)/2$，这里也引申出$M+S$必须是偶数。问题转化为求数组中和为$(M+S)/2$的子列数目，这是一个更简单的DP：$dp[i][s]=dp[i-1][s]+dp[i-1][s-nums[i]]$，表示$nums[0:i]$中子列和为$s$的数目，而且更新规则是右下至左上，因此可以反向更新压缩空间，并减少计算，运行时间优化为4ms。

```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int sum = 0;
        for (auto i : nums) sum += i;
        if (sum < S || -sum > S || (sum + S) % 2 == 1) return 0;
        vector<int> dp((sum + S)/2 + 1, 0);
        dp[0] = 1;
        for (auto n : nums) {
            for (int i = dp.size() - 1; i >= n; i--) {
                dp[i] += dp[i-n];
            }
        }
        return dp.back();
    }
};
```

## 496. Next Greater Element I

用栈来确定下一个较大值。

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> greater;
        stack<int> st;
        for (auto n : nums2) {
            while (!st.empty() && st.top() < n) {
                greater[st.top()] = n;
                st.pop();
            }
            st.push(n);
        }
        vector<int> res;
        for (auto n : nums1) {
            if (greater.find(n) == greater.end()) res.push_back(-1);
            else res.push_back(greater[n]);
        }
        return res;
    }
};
```

## 497. Random Point in Non-overlapping Rectangles

计数所有内部点的数量，而后随机取点，再定位到某个矩形中即可。

```cpp
class Solution {
    vector<vector<int>> r;
    vector<int> lens;
    default_random_engine e;
public:
    Solution(vector<vector<int>>& rects) {
        r = rects;
        for (auto rect : rects) {
            int l = (rect[2]-rect[0]+1)*(rect[3]-rect[1]+1);
            if (lens.empty()) lens.push_back(l);
            else lens.push_back(lens.back() + l);
        }
    }
    
    vector<int> pick() {
        uniform_int_distribution<int> dis(1, lens.back());
        int rand = dis(e);
        int i = lower_bound(lens.begin(), lens.end(), rand) - lens.begin();
        int off = lens[i] - rand;
        int width = r[i][2] - r[i][0] + 1;
        return {r[i][2] - off%width, r[i][3] - off/width};
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(rects);
 * vector<int> param_1 = obj->pick();
 */
```

## 498. Diagonal Traverse

细节题，没啥东西。

```cpp
class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if (matrix.empty() || matrix[0].empty()) return res;
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = 0;
        bool up = true;
        while (res.size() < m * n) {
            res.push_back(matrix[i][j]);
            if (up) {
                if (j + 1 >= n) {
                    i++; up = false;
                } else if (i - 1 < 0) {
                    j++; up = false;
                } else {
                    i--;j++;
                }
            } else {
                if (i + 1 >= m) {
                    j++; up = true;
                } else if (j - 1 < 0) {
                    i++; up = true;
                } else {
                    i++;j--;
                }
            }
        }
        return res;
    }
};
```

## 501. Find Mode in Binary Search Tree

有序的二叉树进行中序遍历可以很容易找出众数，题目关键在于$O(1)$的空间复杂度，使用两次遍历，第一次确定最大出现次数，第二次再进行记录即可。如果考虑到进一步缩减递归栈的空间消耗，可以换作Morris遍历。

```cpp
class Solution {
    int maxCount = 0;
    int curVal = INT_MIN;
    int curCount = 0;
    bool found = false;
    vector<int> mode;
public:
    vector<int> findMode(TreeNode* root) {
        inorder(root);
        found = true;
        curVal = INT_MIN;
        curCount = 0;
        inorder(root);
        return mode;
    }
    
    void inorder(TreeNode* t) {
        if (!t) return;
        inorder(t->left);
        if (curVal == t->val) curCount++;
        else {
            curVal = t->val;
            curCount = 1;
        }
        if (curCount > maxCount) maxCount = curCount;
        if (found && curCount == maxCount) mode.push_back(curVal);
        inorder(t->right);
    }
};
```

Morris中序遍历版本：

```cpp
class Solution {
    int maxCount = 0;
    int curVal = INT_MIN;
    int curCount = 0;
    bool found = false;
    vector<int> mode;
public:
    vector<int> findMode(TreeNode* root) {
        MorrisInorder(root);
        found = true;
        curVal = INT_MIN;
        curCount = 0;
        MorrisInorder(root);
        return mode;
    }
    
    void count(int val) {
        if (curVal == val) curCount++;
        else {
            curVal = val;
            curCount = 1;
        }
        if (curCount > maxCount) maxCount = curCount;
        if (found && curCount == maxCount) mode.push_back(curVal);
    }
    
    void MorrisInorder(TreeNode* t) {
        TreeNode* cur = t, *prev = nullptr;
        while (cur) {
            if (!cur->left) {
                count(cur->val);
                cur = cur->right;
            } else {
                prev = cur->left;
                while (prev->right && prev->right != cur) prev = prev->right;
                if (prev->right) {
                    prev->right = nullptr;
                    count(cur->val);
                    cur = cur->right;
                } else {
                    prev->right = cur;
                    cur = cur->left;
                }
            }
        }
    }
};
```

## 502. IPO

贪心：每次选择可选范围内利润最大的。

```cpp
class Solution {
public:
    int findMaximizedCapital(int k, int W, vector<int>& Profits, vector<int>& Capital) {
        vector<pair<int, int>> pc;
        pc.reserve(Profits.size());
        for (int i = 0; i < Profits.size(); i++) pc.push_back({Profits[i], Capital[i]});
        sort(pc.begin(), pc.end(), [](pair<int, int>& a, pair<int, int>& b) {
            return a.second < b.second;
        });
        priority_queue<pair<int, int>> q;
        int i = 0;
        while (k-- > 0) {
            while (i < pc.size() && pc[i].second <= W) q.push(pc[i++]);
            if (q.empty()) break;
            W += q.top().first;
            q.pop();
        }
        return W;
    }
};
```

## 503. Next Greater Element II

思路同496题。这里不同的是需要两遍扫描，第一遍确定后方的较大数，第二遍确定前方，直到扫描过栈顶元素位置，表示无更大的数了。

```cpp
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> res(nums.size(), -1);
        stack<pair<int, int>> s;
        for (int i = 0; i < nums.size(); i++) {
            while (!s.empty() && s.top().second < nums[i]) {
                res[s.top().first] = nums[i];
                s.pop();
            }
            s.push({i, nums[i]});
        }
        for (int i = 0; i < nums.size(); i++) {
            if (s.empty() || s.top().first <= i) break;
            while (!s.empty() && s.top().second < nums[i] && s.top().first > i) {
                res[s.top().first] = nums[i];
                s.pop();
            }
        }
        return res;
    }
};
```

## 508. Most Frequent Subtree Sum

遍历+记录次数。

```cpp
class Solution {
public:
    vector<int> findFrequentTreeSum(TreeNode* root) {
        unordered_map<int, int> m;
        int maxCount = 0;
        function<int(TreeNode*)> dfs = [&](TreeNode* t) {
            if (!t) return 0;
            int r = dfs(t->left) + dfs(t->right) + t->val;
            if (++m[r] > maxCount) maxCount++;
            return r;
        };
        dfs(root);
        vector<int> res;
        for (auto p : m) if (p.second == maxCount) res.push_back(p.first);
        return res;
    }
};
```

## 513. Find Bottom Left Tree Value

BFS按层遍历，需记录每层节点数量。

```cpp
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        int res = 0, cur = 1, next = 0;
        queue<TreeNode*> q({root});
        while (!q.empty()) {
            res = q.front()->val;
            while (cur-- > 0) {
                auto t = q.front();
                q.pop();
                for (auto tt : {t->left, t->right}) {
                    if (tt) {
                        q.push(tt);
                        next++;
                    }
                }
            }
            cur = next;
            next = 0;
        }
        return res;
    }
};
```

或者DFS，取右子树的最深最左节点仅当其深度大于左子树。

```cpp
class Solution {
    pair<int, int> dfs(TreeNode* t, int depth) {
        auto l = t->left ? dfs(t->left, depth + 1) : make_pair(t->val, depth);
        auto r = t->right ? dfs(t->right, depth + 1) : make_pair(t->val, depth);
        return r.second > l.second ? r : l;
    }
public:
    int findBottomLeftValue(TreeNode* root) {
        return dfs(root, 0).first;
    }
};
```

## 514. Freedom Trail

设$dp[i][j]$为在`ring[j]`，和`key[i]`状态下所需要转动的最小次数，此时`key[i]`还未被按下。因此首先需要转动到`key[i]`，状态转移至`i+1`，设`nextj`是所有`key[i]`在`ring`中的位置，`d(a,b)`表示`ring`从`a`转到`b`所需最短距离，则显然有：
$$
dp[i][j]=\min_{nextj}\{dp[i+1][nextj]+d(j,nextj)+1\}
$$
值得注意的是，本题不需要计算所有的$j$，因为我们每次停留的位置必然是`key[i-1]`所在的位置，因此只要记录这些位置即可。

```cpp
class Solution {
public:
    int findRotateSteps(string ring, string key) {
        vector<vector<int>> dp(key.size() + 1, vector<int>(ring.size(), 0));
        vector<vector<int>> m(26, vector<int>());
        for (int i = 0; i < ring.size(); i++) m[ring[i]-'a'].push_back(i);
        auto d = [&](int j, int nextj) -> int {
            int t = abs(j - nextj);
            return t > ring.size() / 2 ? ring.size() - t : t;
        };
        for (int i = key.size() - 1; i >= 0; i--) {
            if (i == 0) {
                int tmp = INT_MAX;
                for (auto nextj : m[key[i]-'a']) {
                    tmp = min(tmp, dp[i+1][nextj] + d(0, nextj) + 1);
                }
                dp[i][0] = tmp;
                break;
            }
            for (auto j : m[key[i-1]-'a']) {
                int tmp = INT_MAX;
                for (auto nextj : m[key[i]-'a']) {
                    tmp = min(tmp, dp[i+1][nextj] + d(j, nextj) + 1);
                }
                dp[i][j] = tmp;
            }
        }
        return dp[0][0];
    }
};
```

## 515. Find Largest Value in Each Tree Row

一样DFS记录深度或BFS按层遍历，这里只写DFS。

```cpp
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        vector<int> res;
        function<void(TreeNode*, int)> dfs = [&](TreeNode* t, int depth) {
            if (!t) return;
            if (depth >= res.size()) res.push_back(t->val);
            else if (res[depth] < t->val) res[depth] = t->val;
            dfs(t->left, depth + 1);
            dfs(t->right, depth + 1);
        };
        dfs(root, 0);
        return res;
    }
};
```

## 516. Longest Palindromic Subsequence

经典DP。

```cpp
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int k = 0; k < n; k++) {
            for (int i = 0; i + k < n; i++) {
                int j = i + k;
                if (i == j) dp[i][j] = 1;
                else if (s[i] == s[j]) dp[i][j] = dp[i+1][j-1] + 2;
                else dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
            }
        }
        return dp[0][n-1];
    }
};
```

空间优化，上一轮的`dp[i+1][j]`就是`dp[j]`，但是`dp[i+1][j-1]`需要额外保存，否则会被更新。

```cpp
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<int> dp(n, 0);
        for (int i = n-1; i >= 0; i--) {
            int old = 0;
            for (int j = i; j < n; j++) {
                int tmp = dp[j];
                if (j == i) dp[j] = 1;
                else if (s[j] == s[i]) dp[j] = old + 2;
                else if (dp[j-1] > dp[j]) dp[j] = dp[j-1];
                old = tmp;
            }
        }
        return dp[n-1];
    }
};
```

## 517. Super Washing Machines

考虑机器$i$的左侧和右侧机器，设左侧衣服数量与目标数量的差值为$L$，右侧差值为$R$：

1. 如果$L\ge 0$且$R\ge 0$，说明要达到最终的均衡状态，需要将两侧多余的衣服传递给机器$i$，而两侧向中间的传递是可以同时完成的，因此至少需要$max(L,R)$轮。
2. 如果$L<0$且$R<0$，则需要将机器$i$的衣服向两侧传递，这是无法同时完成的，一次只能向一侧传，因此至少需要$(-L-R)$轮。
3. 对于异号的情况，则会有衣服经手机器$i$向某侧传递，以平衡某侧，因此至少需要$max(|L|,|R|)$轮。

我们遍历所有机器$i$，则最终的传递轮次是这些情况中的最大值。同时可以看到，第一种和第三种情况可以用绝对值合并，代码如下：

```cpp
class Solution {
public:
    int findMinMoves(vector<int>& machines) {
        int s = 0;
        for (auto i : machines) s += i;
        if (s % machines.size() != 0) return -1;
        int target = s / machines.size();
        int res = 0, left = 0, right = 0;
        for (auto i : machines) {
            right -= (i - target);
            if (left < 0 && right < 0) res = max(res, -left-right);
            else res = max(res, max(abs(left), abs(right)));
            left += (i - target);
        }
        return res;
    }
};
```

## 518. Coin Change 2

设$dp[n][w]$为在`coins[0:n]`之中组合`w`的总数，则对于$k=w/coins[n]$，显然有：
$$
dp[n][w]=dp[n-1][w]+dp[n-1][w-coins[n]]+...+dp[n-1][w-k\times coins[n]]\\
=\sum_{i=0}^{w/coins[n]}dp[n-1][w-i\times coins[n]]
$$
和式第一个后面的元素加起来恰好是$dp[n][w-coins[n]]$，因此有：
$$
dp[n][w]=dp[n-1][w]+dp[n][w-coins[n]]
$$
边界情况，当$w=0$，则仅存在一种组合（空集），$w<coins[n]$，则后者为0。空间优化：注意到上式每次更新使用左上方的值，因此可以压缩数组到一维。

```cpp
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1, 0);
        dp[0] = 1;
        for (int i = 0; i < coins.size(); i++) {
            for (int j = coins[i]; j <= amount; j++) {
                if (i == 0) dp[j] = j % coins[0] == 0 ? 1 : 0;
                else dp[j] = dp[j] + dp[j - coins[i]];
            }
        }
        return dp[amount];
    }
};
```

## 522. Longest Uncommon Subsequence II

假设最长非公共子序列$s$一定是某个字符串$S_i$的子序列，那么$s$就等于$S_i$，否则，说明$S_i$是某个$S_j$的子序列导致$s$不能增长，那么$s$也成为了$S_j$的子序列，与假设矛盾（$s$是非公共的）。基于这样的观察，本题遍历所有字符串即可。

```cpp
class Solution {
public:
    int findLUSlength(vector<string>& strs) {
        auto isSub = [](const string& a, const string& b) -> bool {
            if (b.size() > a.size()) return false;
            int i = 0;
            for (auto c : a) {
                if (b[i] == c) i++;
            }
            return i == b.size();
        };
        int res = -1;
        for (int i = 0; i < strs.size(); i++) {
            bool ok = true;
            for (int j = 0; j < strs.size(); j++) {
                if (j != i && isSub(strs[j], strs[i])) {
                    ok = false;
                    break;
                }
            }
            if (ok) res = max(res, static_cast<int>(strs[i].size()));
        }
        return res;
    }
};
```

## 523. Continuous Subarray Sum

用哈希表找当前`[0:i]`累加的余数是否等于之前某一次`[0:j]`的累加余数即可，这样`[j+1:i]`就是目标子列。需要考虑除数为零的情况，在此情况下只有连续的0能够满足条件。同时由于0是任何数的倍数，因此在开始前就把0加到哈希表中，以处理开始连续两个零的corner case。加入哈希表的过程需要有间隔，以免子列长度为1。

```cpp
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        if (nums.size() < 2) return false;
        if (k == 0) {
            bool pre = false;
            for (auto num : nums) {
                if (num == 0) {
                    if (pre) return true; 
                    else pre = true;
                } else pre = false;
            }
            return false;
        }
        unordered_set<long long> res({0});
        long long cur = nums[0] % k, pre = nums[0] % k;
        for (int i = 1; i < nums.size(); i++) {
            cur = (cur + nums[i]) % k;
            if (res.find(cur) != res.end()) return true;
            res.insert(pre);
            pre = cur;
        }
        return false;
    }
};
```

## 524. Longest Word in Dictionary through Deleting

遍历。

```cpp
class Solution {
    int match(const string& s, const string& a) {
        if (a.size() > s.size()) return -1;
        int i = 0, j = 0;
        while (i < s.size()) {
            if (s[i++] == a[j]) j++;
            if (j == a.size()) return a.size();
        }
        return -1;
    }
public:
    string findLongestWord(string s, vector<string>& d) {
        int res = -1;
        int maxLen = 0;
        for (int i = 0; i < d.size(); i++) {
            int t = match(s, d[i]);
            if (t > maxLen) {
                maxLen = t;
                res = i;
            } else if (t == maxLen && d[i] < d[res]) {
                res = i;
            }
        }
        return res < 0 ? "" : d[res];
    }
};
```

## 525. Contiguous Array

滚动记录0-1计数的差值，如果和最先一次差值相等，两者索引差为一个子列长。

```cpp
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        int diff = 0;
        unordered_map<int, int> m;
        m[0] = -1;
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == 0) diff++;
            else diff--;
            if (m.find(diff) == m.end()) m[diff] = i;
            else res = max(res, i - m[diff]);
        }
        return res;
    }
};
```

## 526. Beautiful Arrangement

我的解法，DFS+memoization：

```cpp
class Solution {
    int dfs(int used, int index, int N, vector<unordered_map<int, int>>& m) {
        if (index > N) return 1;
        if (m[index].find(used) != m[index].end()) return m[index][used];
        int r = 0;
        for (int i = 1; i <= N; i++) {
            if (!(used & (1 << i)) && ((i % index) == 0 || (index % i) == 0)) {
                r += dfs(used | (1<<i), index + 1, N, m);
            }
        }
        m[index][used] = r;
        return r;
    }
public:
    int countArrangement(int N) {
        int used = 0;
        vector<unordered_map<int, int>> m(N + 1, unordered_map<int, int>());
        return dfs(used, 1, N, m);
    }
};
```

有一个常数时间非常小的深搜解法，对现成的排列进行交换：

```cpp
class Solution {
public:
    int countArrangement(int N) {
        vector<int> candidates(N);
        for(int i = 0; i < N; i++)
            candidates[i] = i + 1;
        return recurse(N, candidates);
    }

private:
    int recurse(int position, vector<int>& candidates){
        if(position == 0) return 1;
        int ret = 0;
        for(int i = 0; i < position; i++){
            if(candidates[i] % position == 0 || position % candidates[i] == 0){
                swap(candidates[i], candidates[position - 1]);
                ret += recurse(position - 1, candidates);
                swap(candidates[i], candidates[position - 1]);
            }
        }
        return ret;
    }
};
```

## 529. Minesweeper

细节题，没啥意思。

```cpp
class Solution {
public:
    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int m = board.size(), n = board[0].size();
        vector<vector<int>> adj({{0, 1},{0,-1},{-1,0},{1,0},{1,1},{1,-1},{-1,-1},{-1,1}});
        function<void(int, int)> dfs = [&](int i, int j) {
            if (board[i][j] == 'M') {
                board[i][j] = 'X';
                return;
            }
            int count = 0;
            for (auto d : adj) {
                int ni = i + d[0], nj = j + d[1];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && board[ni][nj] == 'M')
                    count++;
            }
            if (count != 0) board[i][j] = '0'+count;
            else {
                board[i][j] = 'B';
                for (auto d : adj) {
                    int ni = i + d[0], nj = j + d[1];
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n && board[ni][nj] == 'E')
                        dfs(ni, nj);
                }
            }
        };
        dfs(click[0], click[1]);
        return board;
    }
};
```

## 538. Convert BST to Greater Tree

反向遍历。

```cpp
class Solution {
    void dfs(TreeNode* t, int& sum) {
        if (!t) return;
        dfs(t->right, sum);
        sum += t->val;
        t->val = sum;
        dfs(t->left, sum);
    }
public:
    TreeNode* convertBST(TreeNode* root) {
        int sum = 0;
        dfs(root, sum);
        return root;
    }
};
```

## 540. Single Element in a Sorted Array

在满足题目条件的连续子序列中，单个出现的元素必然在长度为奇数的子序列中。通过二分查找，根据两边的长度奇偶性来确定范围。

```cpp
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int l = 0, r = nums.size();
        if (r == 1) return nums[0];
        while (r - l > 1) {
            int n = (r - l) / 2;
            int mid = l + n;
            if (nums[mid] == nums[mid+1]) {
                if (n % 2 == 0) l = mid + 2;
                else r = mid;
            } else if (nums[mid] == nums[mid-1]) {
                if (n % 2 == 0) r = mid - 1;
                else l = mid + 1;
            } else return nums[mid];
        }
        return nums[l];
    }
};
```

## 542. 01 Matrix

首先从0的点开始，进行广搜，一层一层把最短距离传递下去直到所有点都更新到。

```cpp
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> r(m, vector<int>(n, -1));
        queue<pair<int, int>> q;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    r[i][j] = 0;
                    q.push({i, j});
                }
            }
        }
        while (!q.empty()) {
            int i = q.front().first, j = q.front().second;
            q.pop();
            if (i + 1 < m && r[i+1][j] < 0) {
                r[i+1][j] = r[i][j] + 1;
                q.push({i+1, j});
            }
            if (j + 1 < n && r[i][j+1] < 0) {
                r[i][j+1] = r[i][j] + 1;
                q.push({i, j+1});
            }
            if (i - 1 >= 0 && r[i-1][j] < 0) {
                r[i-1][j] = r[i][j] + 1;
                q.push({i-1, j});
            }
            if (j - 1 >= 0 && r[i][j-1] < 0) {
                r[i][j-1] = r[i][j] + 1;
                q.push({i, j-1});
            }
        }
        return r;
    }
};
```

写完后看到另一个有趣的解法是从相反的方向更新距离表两次就能够覆盖所有情况。

```cpp
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> r(m, vector<int>(n, 1000000));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    r[i][j] = 0;
                } else {
                    if (i > 0) r[i][j] = min(r[i][j], r[i-1][j] + 1);
                    if (j > 0) r[i][j] = min(r[i][j], r[i][j-1] + 1);
                }
            }
        }
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (i + 1 < m) r[i][j] = min(r[i][j], r[i+1][j] + 1);
                if (j + 1 < n) r[i][j] = min(r[i][j], r[i][j+1] + 1);
            }
        }
        return r;
    }
};
```

## 543. Diameter of Binary Tree

遍历左右子树寻找最深路径。

```cpp
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int res = 0;
        function<int(TreeNode*)> dfs = [&](TreeNode* t) {
            if (!t) return 0;
            int l = dfs(t->left);
            int r = dfs(t->right);
            res = max(res, l + r);
            return max(l + 1, r + 1);
        };
        dfs(root);
        return res;
    }
};
```

## 546. Remove Boxes

只提供下标`[i,j]`不足以描述子问题，在DP表中还需要记录可能与其结合消去的外部的左侧/右侧元素，由于对称性，只需要一侧就可以。因此用`dp[i][j][k]`表示`[i,j]`之间的盒子以及左侧可能的`k`个与`boxes[i]`同色的盒子一同消去可以得到的最高分。显然，如果`i=j`，则$dp[i][i][k]=(k+1)^2$，考虑`i<j`的情况，此时左侧`k`个元素可以选择直接和`i`消去，那么：
$$
dp[i][j][k]=(k+1)^2+dp[i+1][j][0]
$$
也可以选择先消去`[i+1,m-1]`中的所有元素，然后与`i`共同作为后续区间`[m,j]`的左侧元素参与消除，此时必须满足`i`与`m`同色：
$$
dp[i][j][k]=\max_{box[i]=box[m]}\{dp[i+1][m-1][0]+dp[m][j][k+1]\}
$$

```cpp
class Solution {
public:
    int removeBoxes(vector<int>& boxes) {
        int n = boxes.size();
        int dp[100][100][100] = {0};
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                for (int k = 0; k <= i; k++) {
                    if (i == j) dp[i][j][k] = (k + 1)*(k + 1);
                    else {
                        dp[i][j][k] = (k + 1)*(k + 1) + dp[i+1][j][0];
                        for (int m = i + 1; m <= j; m++) {
                            if (boxes[m] == boxes[i]) {
                                dp[i][j][k] = max(dp[i][j][k], dp[i+1][m-1][0]+dp[m][j][k+1]);
                            }
                        }
                    }
                }
            }
        }
        return dp[0][n-1][0];
    }
};
```

上面的解法完全正确，但是粒度太小导致速度慢，实际上可以将连续相同的盒子事先整合起来，因为已经连续的盒子肯定是一起消除得分最高，不可能再分开了。用一个pair数组来保存，在其之上运行自顶向下的dp：

```cpp
class Solution {
public:
    int removeBoxes(vector<int>& boxes) {
        vector<pair<int, int>> c({{boxes[0], 0}});
        for (auto i : boxes) {
            if (i == c.back().first) c.rbegin()->second++;
            else c.push_back({i, 1});
        }
        int n = c.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(100, -1)));
        function<int(int, int, int)> dfs = [&](int i, int j, int k) {
            if (dp[i][j][k] >= 0) return dp[i][j][k];
            if (i > j) return dp[i][j][k] = 0;
            if (i == j) return dp[i][j][k] = (k + c[i].second)*(k + c[i].second);
            dp[i][j][k] = (k + c[i].second)*(k + c[i].second) + dfs(i+1, j, 0);
            for (int m = i + 1; m <= j; m++) {
                if (c[i].first == c[m].first) {
                    dp[i][j][k] = max(dp[i][j][k], dfs(i+1, m-1, 0) + dfs(m, j, k+c[i].second));
                }
            }
            return dp[i][j][k];
        };
        return dfs(0, n-1, 0);
    }
};
```

## 547. Friend Circles

就是求无向图中的连通分量数，直接深搜就可。

```cpp
class Solution {
public:
    int findCircleNum(vector<vector<int>>& M) {
        int res = 0, n = M.size();
        vector<bool> visited(n, false);
        function<void(int)> dfs = [&](int i) {
            visited[i] = true;
            for (int j = 0; j < n; j++) {
                if (M[i][j] == 1 && !visited[j]) dfs(j);
            }
        };
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs(i);
                res++;
            }
        }
        return res;
    }
};
```

## 552. Student Attendance Record II

本题DP，设$dp[n][A][L]$为长度为$n$的记录个数，其中至多有$A$个A，以及该记录左边有紧临的$L$个L。那么在第一个位置可以尝试放置$A,L,P$，从而很容易导出递推式：
$$
dp[n][A][L]=dp[n-1][A-1][0]+dp[n-1][A][0]+dp[n-1][A][L+1]
$$
在本题限制条件下，$A$只可取值$0,1$，$L$取值为$0,1,2$，因此上式在$A=0$和$L=2$时需要减少两项。实现时不需要三维矩阵，只需要保持前一轮的计算结果即可。

```cpp
class Solution {
public:
    int checkRecord(int n) {
        int dp[2][2][3];
        dp[0][0][0] = 2; dp[0][0][1] = 2; dp[0][0][2] = 1;
        dp[0][1][0] = 3; dp[0][1][1] = 3; dp[0][1][2] = 2;
        int pre = 0, cur = 1, mod = 1000000007;
        for (int i = 2; i <= n; i++) {
            for (int A = 0; A < 2; A++) {
                for (int L = 0; L < 3; L++) {
                    dp[cur][A][L] = dp[pre][A][0];
                    if (A > 0) dp[cur][A][L] = (dp[cur][A][L] + dp[pre][A-1][0]) % mod;
                    if (L < 2) dp[cur][A][L] = (dp[cur][A][L] + dp[pre][A][L+1]) % mod;
                }
            }
            swap(pre, cur);
        }
        return dp[pre][1][0];
    }
};
```

## 553. Optimal Division

脑筋急转弯题，答案是固定的。

```cpp
class Solution {
public:
    string optimalDivision(vector<int>& nums) {
        if (nums.size() == 1) return to_string(nums[0]);
        if (nums.size() == 2) return to_string(nums[0]) + "/" + to_string(nums[1]);
        string res = to_string(nums[0]) + "/(";
        for (int i = 1; i < nums.size(); i++) {
            res += to_string(nums[i]);
            if (i != nums.size() - 1) res += "/";
        }
        res += ")";
        return res;
    }
};
```

## 554. Brick Wall

用优先队列存每行目前为止的最右位置，选其中最小的作为切入点，并更新队列，设有$n$块砖，则每块砖进/出队列一次，复杂度为$O(n\log n)$。

```cpp
class Solution {
public:
    int leastBricks(vector<vector<int>>& wall) {
        int height = wall.size();
        vector<int> next(height, 1);
        auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
            return a.first > b.first;
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> q(cmp);
        int res = height;
        for (int i = 0; i < height; i++) {
            q.push(make_pair(wall[i][0], i));
        }
        while (true) {
            int count = 0, t = q.top().first;
            while (!q.empty() && q.top().first == t) {
                int i = q.top().second;
                if (next[i] < wall[i].size()) {
                    q.push(make_pair(q.top().first + wall[i][next[i]], i));
                    next[i]++;
                }
                q.pop();
                count++;
            }
            if (q.empty()) break;
            res = min(res, height - count);
        }
        return res;
    }
};
```

实际上优先队列不需要，我们只需要记录每一个独立的右边界出现的次数，那么擦着这个边界切入就能减少碰撞，因此用哈希表来存储，每块砖计算一次右边界，复杂度为$O(n)$。

```cpp
class Solution {
public:
    int leastBricks(vector<vector<int>>& wall) {
        int height = wall.size(), res = height;
        unordered_map<int, int> m;
        for (int i = 0; i < height; i++) {
            int t = 0;
            for (int j = 0; j < wall[i].size() - 1; j++) {
                t += wall[i][j];
                m[t]++;
                if (height - m[t] < res) res = height - m[t];
            }
        }
        return res;
    }
};
```

## 556. Next Greater Element III

从右向左寻找第一个破坏递减的数字，将其比它大的最右的数字交换。注意交换后，其后面的数字呈递减排列，而此时该位变大以后，后面的数字应按照最小排列，因此倒置后面的数字即可，如下例。这实际上就是寻找下一个字典序排列的算法。

```
186(4751)-->186(5741)-->1865(147)
    | |          |||         |||
```

```cpp
class Solution {
public:
    int nextGreaterElement(int n) {
        string nums = to_string(n);
        int pre = 0;
        for (int i = nums.size() - 1; i >= 0; i--) {
            if (nums[i] >= pre) {
                pre = nums[i];
            } else {
                int j = i + 1;
                for (; j < nums.size(); j++) {
                    if (nums[j] <= nums[i]) break;
                }
                swap(nums[i], nums[j-1]);
                reverse(nums.begin() + i + 1, nums.end());
                pre = -1;
                break;
            }
        }
        if (pre >= 0) return -1;
        long num = stol(nums);
        if (num > INT_MAX) return -1;
        else return num;
    }
};
```

## 557. Reverse Words in a String III

简单题。

```cpp
class Solution {
public:
    string reverseWords(string s) {
        auto pre = s.begin();
        for (auto i = s.begin(); ; i++) {
            if (i == s.end() || *i == ' ') {
                reverse(pre, i);
                if (i == s.end()) break;
                pre = i + 1;
            }
        }
        return s;
    }
};
```

## 558. Logical OR of Two Binary Grids Represented as Quad-Trees

直接分治。

```cpp
/*
// Definition for a QuadTree node.
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;
    
    Node() {
        val = false;
        isLeaf = false;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};
*/
class Solution {
    Node* tleaf = new Node(true, true, nullptr, nullptr, nullptr, nullptr);
    Node* fleaf = new Node(false, true, nullptr, nullptr, nullptr, nullptr);
public:
    Node* intersect(Node* quadTree1, Node* quadTree2) {
        if (!quadTree1->isLeaf && !quadTree2->isLeaf) {
            auto tl = intersect(quadTree1->topLeft, quadTree2->topLeft);
            auto tr = intersect(quadTree1->topRight, quadTree2->topRight);
            auto bl = intersect(quadTree1->bottomLeft, quadTree2->bottomLeft);
            auto br = intersect(quadTree1->bottomRight, quadTree2->bottomRight);
            if (tl->isLeaf && tr->isLeaf && bl->isLeaf && br->isLeaf 
                && tl->val == tr->val && bl->val == br->val && bl->val == tl->val)
                return tl->val ? tleaf : fleaf;
            else
                return new Node(false, false, tl, tr, bl, br);
        } else if (quadTree1->isLeaf) {
            if (quadTree1->val) return quadTree1;
            else return quadTree2;
        } else {
            if (quadTree2->val) return quadTree2;
            else return quadTree1;
        }
    }
};
```

## 559. Maximum Depth of N-ary Tree

简单题。

```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/
class Solution {
public:
    int maxDepth(Node* root) {
        int res = 0;
        function<void(Node*, int)> dfs = [&](Node* t, int depth) {
            if (!t) return;
            res = max(res, depth);
            for (auto p : t->children) {
                dfs(p, depth + 1);
            }
        };
        dfs(root, 1);
        return res;
    }
};
```

## 560. Subarray Sum Equals K

哈希表保存阶段和。

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> m;
        int sum = 0, res = 0;
        m[0] = 1;
        for (auto n : nums) {
            sum += n;
            if (m.find(sum - k) != m.end()) res += m[sum - k];
            m[sum]++;
        }
        return res;
    }
};
```

## 561. Array Partition I

排序。

```cpp
class Solution {
public:
    int arrayPairSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = 0; i < nums.size(); i += 2) res += nums[i];
        return res;
    }
};
```

## 563. Binary Tree Tilt

Traverse.

```cpp
class Solution {
public:
    int findTilt(TreeNode* root) {
        int res = 0;
        function<int(TreeNode*)> dfs = [&](TreeNode* t) {
            if (!t) return 0;
            int left = dfs(t->left);
            int right = dfs(t->right);
            res += abs(left - right);
            return left + right + t->val;
        };
        dfs(root);
        return res;
    }
};
```

## 564. Find the Closest Palindrome

基本思路是获得最接近的较小和较大的回文数并比较。若要获得较大的回文数，尝试将左半部分拷贝到右半部分，如果已经大于原来的数则直接返回，否则将左半部分加1再拷贝，加1会可能溢出，此时要注意进位的问题。另一半问题同理，减1要注意可能会少一位，此时需要将中间部分置为9。

```cpp
class Solution {
    int compare(const string& n) {
        int i = n.size() / 2, j = n.size() / 2;
        if (n.size() % 2 == 0) i--;
        while (i >= 0 && j < n.size()) {
            if (n[i] > n[j]) return 1;
            if (n[i] < n[j]) return -1;
            i--;
            j++;
        }
        return 0;
    }
    string getLargerOne(const string& n) {
        int len = n.size() % 2 == 1 ? n.size() / 2 + 1 : n.size() / 2;
        string r(n);
        if (compare(n) <= 0) {// if it's like 1889, we need 18-->19
            int carry = 1, i = len - 1;
            while (carry) {
                if (i < 0) {
                    r.insert(0, 1, '1');
                    carry = 0;
                } else if (r[i] + carry > '9') {
                    r[i] = r[i] + carry - 10;
                } else {
                    r[i] += carry;
                    carry = 0;
                }
                i--;
            }
        }
        int i = 0, j = r.size() - 1;
        while (i <= j) {
            r[j--] = r[i++];
        }
        return r;
    }
    string getSmallerOne(const string& n) {
        int len = n.size() % 2 == 1 ? n.size() / 2 + 1 : n.size() / 2;
        string r(n);
        if (compare(n) >= 0) {// same idea
            int carry = 1, i = len - 1;
            while (carry) {
                if (r[i] == '0') {
                    r[i] = '9';
                } else {
                    r[i]--;
                    carry = 0;
                }
                i--;
            }
            if (r[0] == '0') {
                r.erase(r.begin());
                r[len-1] = '9';// for something like 1000, we need 990 not 990
            }
        }
        int i = 0, j = r.size() - 1;
        while (i <= j) {
            r[j--] = r[i++];
        }
        return r;
    }
public:
    string nearestPalindromic(string n) {
        if (n.size() == 1) return string(1, n[0]-1);
        string l = getLargerOne(n);
        string s = getSmallerOne(n);
        long ll = stol(l);
        long ss = stol(s);
        long num = stol(n);
        if (num - ss <= ll - num) return s;
        else return l;
    }
};
```

## 565. Array Nesting

每个数只出现一次，意味着图中只存在多个环，且每个数字在某个环中。深搜找最长的环即可。

```cpp
class Solution {
public:
    int arrayNesting(vector<int>& nums) {
        int res = 0;
        function<int(int)> dfs = [&](int i) {
            if (nums[i] == -1) return 0;
            else {
                int t = nums[i];
                nums[i] = -1;
                return 1 + dfs(t);
            }
        };
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != -1) res = max(res, dfs(i));
        }
        return res;
    }
};
```

## 567. Permutation in String

扫描记录字符出现次数，我的解法窗口大小会变化，也可以固定窗口横向扫描同时比对次数。

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        vector<int> count(26, -1);
        for (auto c : s1) {
            if (count[c-'a'] < 0) count[c-'a'] = 1;
            else count[c-'a']++;
        }
        int len = 0;
        for (int i = 0; i < s2.size(); i++) {
            if (count[s2[i]-'a'] < 0) {
                for (int j = i - len; j < i; j++) count[s2[j]-'a']++;
                len = 0;
            } else if (count[s2[i]-'a'] == 0) {
                for (int j = i - len; j < i; j++) {
                    len--;
                    count[s2[j]-'a']++;
                    if (s2[j] == s2[i]) break;
                }
                count[s2[i]-'a']--;
                len++;
            } else {
                count[s2[i]-'a']--;
                len++;
                if (len == s1.size()) return true;
            }
        }
        return false;
    }
};
```

## 572. Subtree of Another Tree

遍历S，判断T是否与当前S的子树等同。

```cpp
class Solution {
    bool check(TreeNode* s, TreeNode* t) {
        if (!s && !t) return true;
        if (!s || !t) return false;
        if (s->val == t->val) return check(s->left, t->left) && check(s->right, t->right);
        return false;
    }
public:
    bool isSubtree(TreeNode* s, TreeNode* t) {
        if (!t) return true;
        if (!s) return false;
        if (check(s, t)) return true;
        return isSubtree(s->left, t) || isSubtree(s->right, t);
    }
};
```

##  576. Out of Boundary Paths

DP，很简单。

```cpp
class Solution {
public:
    int findPaths(int m, int n, int N, int i, int j) {
        if (N == 0) return 0;
        int pre = 0, cur = 1, res = 0, x = i, y = j, mod = 1000000007;
        vector<vector<vector<int>>> dp(2, vector<vector<int>>(m, vector<int>(n, 0)));
        for (int i = 0; i < m; i++) {
            dp[0][i][0] += 1;
            dp[0][i][n-1] += 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][0][j] += 1;
            dp[0][m-1][j] += 1;
        }
        res += dp[pre][x][y];
        for (int k = 1; k < N; k++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    dp[cur][i][j] = 0;
                    if (i - 1 >= 0) dp[cur][i][j] = (dp[cur][i][j] + dp[pre][i-1][j]) % mod;
                    if (j - 1 >= 0) dp[cur][i][j] = (dp[cur][i][j] + dp[pre][i][j-1]) % mod;
                    if (i + 1 < m) dp[cur][i][j] = (dp[cur][i][j] + dp[pre][i+1][j]) % mod;
                    if (j + 1 < n) dp[cur][i][j] = (dp[cur][i][j] + dp[pre][i][j+1]) % mod;
                }
            }
            swap(cur, pre);
            res = (res + dp[pre][x][y]) % mod;
        }
        return res;
    }
};
```

## 581. Shortest Unsorted Continuous Subarray

笨方法，依据这样的事实：能够不用重排的数一定等于前面所有数的最大值和后面所有数的最小值，把阶段最小和最大值存起来，从两边开始通过与最值比较来缩小范围。这样虽然时间复杂度是$O(n)$，但是空间复杂度也是$O(n)$。

```cpp
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int n = nums.size();
        vector<int> maxs(n, 0);
        vector<int> mins(n, 0);
        for (int i = 0; i < nums.size(); i++) {
            if (i == 0) {
                maxs[0] = nums[0];
                mins[n-i-1] = nums[n-i-1];
            } else {
                maxs[i] = max(nums[i], maxs[i-1]);
                mins[n-i-1] = min(nums[n-i-1], mins[n-i]);
            }
        }
        int i = 0, j = n - 1;
        while (i < j) {
            if (mins[i] == nums[i]) i++;
            else break;
        }
        while (j >= i) {
            if (maxs[j] == nums[j]) j--;
            else break;
        }
        return j - i + 1;
    }
};
```

如何优化掉空间复杂度呢？实际上从右开始最左的不需要重排的数就是最左的等于其左边所有数（包括自己）的最大值的数，因此可以从左向右扫描来更新右边界，最小值同理。

```cpp
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int n = nums.size(), maxN = INT_MIN, minN = INT_MAX;
        int end = -1, start = 0;
        for (int i = 0; i < n; i++) {
            maxN = max(maxN, nums[i]);
            if (maxN != nums[i]) end = i;
        }
        for (int i = n - 1; i >= 0; i--) {
            minN = min(minN, nums[i]);
            if (minN != nums[i]) start = i;
        }
        return end - start + 1;
    }
};
```



## 587. Erect the Fence

凸包，Graham扫描法，不同的是题目把与左下角点的共线点也包含了，因此额外做了一点处理。

```cpp
class Solution {
    int cross(int x1, int y1, int x2, int y2) {
        return x1*y2 - x2*y1;
    }
    bool check(vector<int>& a, vector<int>& b, vector<int>& c) {
        return cross(b[0]-a[0], b[1]-a[1], c[0]-b[0], c[1]-b[1]) < 0;
    }
public:
    vector<vector<int>> outerTrees(vector<vector<int>>& points) {
        if (points.size() <= 3) return points;
        auto it = points.begin();
        for (auto i = points.begin(); i != points.end(); i++) {
            if ((*i)[1] < (*it)[1]) it = i;
            else if ((*i)[1] == (*it)[1] && (*i)[0] < (*it)[0]) it = i;
        }
        swap(*it, *points.begin());
        int x0 = points[0][0], y0 = points[0][1];
        sort(points.begin() + 1, points.end(), [&](const vector<int>& p1, const vector<int>& p2) {
            int x1 = p1[0]-x0, y1 = p1[1]-y0, x2 = p2[0]-x0, y2 = p2[1]-y0;
            int c = cross(x1, y1, x2, y2);
            if (c > 0) return true;
            else if (c < 0) return false;
            else return abs(x1) > abs(x2) || (abs(x1) == abs(x2) && abs(y1) > abs(y2));
        });
        int t = points.size();
        for (int i = 2; i <= points.size(); i++) {
            if (cross(points[1][0]-x0, points[1][1]-y0, points[i][0]-x0,points[i][1]-y0) != 0) {
                t = i;
                break;
            }
        }
        sort(points.begin() + 1, points.begin() + t, [&](const vector<int>& p1, const vector<int>& p2) {
            return abs(p1[0] - x0) < abs(p2[0] - x0) || 
                (abs(p1[0] - x0) == abs(p2[0] - x0) && abs(p1[1] - y0) < abs(p2[1] - y0));
        });
        vector<vector<int>> res(points.begin(), points.begin() + 3);
        for (int i = 3; i < points.size(); i++) {
            while (check(*(res.rbegin()+1), *res.rbegin(), points[i])) {
                res.pop_back();
            }
            res.push_back(points[i]);
        }
        return res;
    }
};
```

## 593. Valid Square

通过向量加法判断平行四边形，再乘法判断直角，随后判断边长是否相等且非零。

```cpp
class Solution {
    vector<int> add(vector<int>& p1, vector<int>& p2) {
        return {p1[0]+p2[0], p1[1]+p2[1]};
    }
    vector<int> sub(vector<int>& p1, vector<int>& p2) {
        return {p1[0]-p2[0], p1[1]-p2[1]};
    }
    bool equal(vector<int>&& p1, vector<int>& p2) {
        return p1[0]==p2[0] && p1[1]==p2[1];
    }
    int mul(vector<int>& p1, vector<int>& p2) {
        return p1[0]*p2[0]+p1[1]*p2[1];
    }
    int slen(vector<int>& p1) {
        return p1[0]*p1[0]+p1[1]*p1[1];
    }
public:
    bool validSquare(vector<int>& p1, vector<int>& p2, vector<int>& p3, vector<int>& p4) {
        auto p12 = sub(p2, p1);
        auto p13 = sub(p3, p1);
        auto p14 = sub(p4, p1);
        if (equal(add(p12, p13), p14)) {
            return mul(p12, p13) == 0 && slen(p12) == slen(p13) && slen(p12) > 0;
        } else if (equal(add(p13, p14), p12)) {
            return mul(p13, p14) == 0 && slen(p13) == slen(p14) && slen(p13) > 0;
        } else if (equal(add(p12, p14), p13)) {
            return mul(p12, p14) == 0 && slen(p12) == slen(p14) && slen(p12) > 0;
        } else return false;
    }
};
```

## 594. Longest Harmonious Subsequence

哈希表计数。也可以直接排序，省空间。

```cpp
class Solution {
public:
    int findLHS(vector<int>& nums) {
        unordered_map<int, int> m;
        int res = 0;
        for (auto n : nums) {
            m[n]++;
            if (m.find(n-1) != m.end()) res = max(res, m[n-1]+m[n]);
            if (m.find(n+1) != m.end()) res = max(res, m[n+1]+m[n]);
        }
        return res;
    }
};
```

## 598. Range Addition II

求交集。

```cpp
class Solution {
public:
    int maxCount(int m, int n, vector<vector<int>>& ops) {
        int xmin = m, ymin = n;
        for (auto op : ops) {
            xmin = min(xmin, op[0]);
            ymin = min(ymin, op[1]);
        }
        return xmin*ymin;
    }
};
```

## 599. Minimum Index Sum of Two Lists

哈希。

```cpp
class Solution {
public:
    vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
        unordered_map<string, int> m;
        for (int i = 0; i < list1.size(); i++) {
            m[list1[i]] = -i;
        }
        int minIndex = INT_MAX;
        for (int i = 0; i < list2.size(); i++) {
            if (m.find(list2[i]) != m.end()) {
                m[list2[i]] = i - m[list2[i]];
                minIndex = min(m[list2[i]], minIndex);
            }
        }
        vector<string> res;
        for (auto p : m) {
            if (p.second == minIndex) res.push_back(p.first);
        }
        return res;
    }
};
```



## 600. Non-negative Integers without Consecutive Ones

首先考虑$[0,(11...1)_k]$区间内有多少个满足条件的数？实际上考虑最高位，如果最高位取1，则次高位必然取0，剩余$k-2$位任意取；如果最高位取0，则剩余$k-1$位任意取。因此该递推式满足斐波那契数列：$f[k]=f[k-1]+f[k-2]$。可以很容易计算出32位下的这些值。

之后考虑一个具体的数$N=a_na_{n-1}...a_1$，从最高位$a_n$看，它必然是1，因此如果将此位设置为0，剩余$n-1$位任意取（即为$f[n-1]$），都不大于$N$，如果此位不变，我们迭代考虑不大于$a_{n-1}a_{n-2}...a_1$的数，此时如果$a_{n-1}$为零，则不需要干什么，继续找下一个非零位即可，但如果$a_{n-1}$为1，因为我们不能保留连续的1，必须将其设置为0，任意取余下的位，但此时不能再向下迭代了，因为已经出现了连续的1，直接将结果返回即可。按上述思路迭代即可。

```cpp
class Solution {
public:
    int findIntegers(int num) {
        int f[32] = {1, 2};
        for (int i = 2; i < 32; i++) f[i] = f[i-1] + f[i-2];
        int pre = 0, res = 0;
        for (int k = 31; k >= 0; k--) {
            if ((1 << k) & num) {
                res += f[k];
                if (pre) return res;
                pre = 1;
            } else {
                pre = 0;
            }
        }
        return res + 1;
    }
};
```

## 611. Valid Triangle Number

我们可以首先对数组进行排序，按升序排列后，假如我们确定了最短和最长的边`lo`和`hi`，则最后一条边必然在`(lo,hi)`中确定，且需要满足的条件只剩`mid>hi-lo`，其余条件均由顺序关系确定了。而此时区间内的边长是有序的，可以通过二分查找确定`mid`的最小值。并且由于每次最长边`hi`后移，只会变长，从而`hi-lo`是变大的，`mid`的最小值不会变小，因此查找区间的起点不用重置为`lo+1`。

```cpp
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        if (nums.size() <= 2) return 0;
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int lo = 0; lo < nums.size() - 2; lo++) {
            auto it = nums.begin() + lo + 1;
            for (int hi = lo + 2; hi < nums.size(); hi++) {
                it = upper_bound(it, nums.begin()+hi, nums[hi]-nums[lo]);
                res += nums.begin() + hi - it;
            }
        }
        return res;
    }
};
```

这一解法复杂度为$O(n^2\log n)$，但实际上这里用二分查找是不需要的（反而加大了复杂度），由于每次`mid`的最小值都在增大，就算我们直接线性从`lo`开始扫描，寻找最小的`mid`，最终也最多把整个数组扫描一遍，从而在第二个循环中最多对数组扫描两遍（hi从lo+2至最后，k从lo+1到最后），达到$O(n^2)$的时间复杂度：

```cpp
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        if (nums.size() <= 2) return 0;
        sort(nums.begin(), nums.end());
        int res = 0;
        for (int lo = 0; lo < nums.size() - 2; lo++) {
            int k = lo + 1;
            for (int hi = lo + 2; hi < nums.size(); hi++) {
                while (k < hi && nums[k] <= nums[hi] - nums[lo]) k++;
                res += hi - k;
            }
        }
        return res;
    }
};
```

## 621. Task Scheduler

一个智商被碾压的问题。

```cpp
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        unordered_map<char,int> mp;
        int count = 0;
        for(auto e : tasks) {
            mp[e]++;
            count = max(count, mp[e]);
        }
        int ans = (count-1)*(n+1);
        for(auto e : mp) if (e.second == count) ans++;
        return max((int)tasks.size(), ans);
    }
};
```

一个容易理解的贪心调度策略：每轮调度n+1个，优先调度当前剩余长度最长的任务。用优先队列保存剩余任务长度。

```cpp
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        int counter[26] = {0};
        for (char task : tasks) {
            counter[task - 'A']++;
        }
        priority_queue<int> pq;
        for (int i = 0; i < 26; i++) {
            if (counter[i]) {
                pq.push(counter[i]);
            }
        }
        int time = 0;
        while (!pq.empty()) {
            vector<int> remaining;
            int all = n + 1;
            while (all && !pq.empty()) {
                int counts = pq.top();
                pq.pop();
                if (--counts) {
                    remaining.push_back(counts);
                }
                time++;
                all--;
            }
            for (int counts : remaining) {
                pq.push(counts);
            }
            if (pq.empty()) {
                break;
            }
            time += all;
        }
        return time;
    }
};
```

## 629. K Inverse Pairs Array

DP，设$f(n,k)$为$[1,n]$的整数构成的$k$逆序排列的数量，我们考虑排列中最大的数$n$的位置，如果它在倒数第$i$个位置：
$$
A_1A_2...A_{n-i}N...A_n
$$
则它后面的数均比它小，产生了$i-1$个逆序对，它前面的数均比他小，不产生逆序对。此时包含$N$的逆序对为$i-1$，将$N$从排列中删去，我们只需计算剩余$n-1$个数产生$k-i+1$个逆序对的排列数目即可。从而递推式如下：
$$
f(n,k)=\sum_{i=1}^{n}f(n-1,k-i+1)
$$
同时需要注意，如果计算过程中遇到$k-i+1<0$，说明$n$在该位置已产生超量的逆序对，直接为零即可。同时将$f(n,k-1)$展开与上式比对后，可以化简为：
$$
f(n,k)=f(n-1,k)+f(n,k-1)-f(n-1,k-n)
$$
时间复杂度为$O(kn)$。

```cpp
class Solution {
public:
    int kInversePairs(int n, int k) {
        if (k > n*(n-1)/2) return 0;
        int mod = 1000000007;
        vector<vector<long>> dp(n + 1, vector<long>(k + 1, 0));
        dp[1][0] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < k + 1 && j <= i*(i-1)/2; j++) {
                if (j == 0) dp[i][j] = 1;
                else {
                    dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % mod;
                    if (j >= i) dp[i][j] = (dp[i][j] - dp[i-1][j-i] + mod) % mod;
                }
            }
        }
        return dp[n][k];
    }
};
```

## 630. Course Schedule III

本题可以用DP来做但是sadly超时了，所以还是贪心。贪心策略如下：将任务按照截止时间升序排列，每次选择一个任务，如果该任务能够在截止时间前完成，则直接选择，否则从已选任务中选择一个持续时间最长的任务，如果它比此任务长，则替换之，否则不选择这个任务。

实际上基于这样的考虑，首先，对于任务集$J$，开始时间$T$越早，能完成的任务数越多。现在我们选择到第$K$个任务，且之前的1至K-1个任务的选取是最优的，如果它比之前所有任务都长，且不能在截止时间内完成，则它必然不在最优选择中，因为如果选择它的话，后续任务的开始时间必然晚于当前的开始时间。反之，如果它短于之前的最长任务，则替换后，后续任务开始时间早于当前时间，且当前的1至K个任务完成数量不减，策略更优。

```cpp
class Solution {
public:
    int scheduleCourse(vector<vector<int>>& courses) {
        sort(courses.begin(), courses.end(), [](vector<int>& a, vector<int>& b) {
            return a[1] < b[1]; 
        });
        priority_queue<int> pq;
        int start = 0, res = 0;
        for (auto course : courses) {
            if (start + course[0] <= course[1]) {
                res++;
                start += course[0];
                pq.push(course[0]);
            } else if (!pq.empty() && pq.top() > course[0]) {
                start += course[0] - pq.top();
                pq.pop();
                pq.push(course[0]);
            }
        }
        return res;
    }
};
```

## 632. Smallest Range Covering Elements from K Lists

维护k个数组的头部元素，其最大和最小值构成的区间是当前头部最小区间。每次迭代将最小元素去除，向后移一位即可遍历所有可能的最小区间。若共有$n$个元素，分成$k$个数组，则用红黑树/最小堆来存k个头部元素进行插入删除的话，复杂度为$n\lg k$。

```cpp
class Solution {
public:
    vector<int> smallestRange(vector<vector<int>>& nums) {
        vector<int> res({0, INT_MAX});
        multiset<tuple<int, int, int>> s;
        int s_min, i, j, s_max = INT_MIN;
        for (int i = 0; i < nums.size(); i++) {
            s.insert({nums[i][0], i, 0});
            s_max = max(s_max, nums[i][0]);
        }
        while (true) {
            tie(s_min, i, j) = *s.begin();
            if (s_max - s_min < res[1] - res[0]) {
                res[0] = s_min;
                res[1] = s_max;
            }
            s.erase(s.begin());
            for (j = j + 1; j < nums[i].size(); j++) {
                if (nums[i][j] > s_min) {
                    s.insert({nums[i][j], i, j});
                    s_max = max(s_max, nums[i][j]);
                    break;
                }
            }
            if (s.size() < nums.size()) break;
        }
        return res;
    }
};
```



## 636. Exclusive Time of Functions

一个栈保存start point，添加一项表示这个点之后执行完的其他任务总时间，在最终计算时候减去它，避免重复计算时间。

```cpp
class Solution {
public:
    vector<int> exclusiveTime(int n, vector<string>& logs) {
        vector<int> res(n, 0);
        stack<tuple<int, int, int>> s;// id, start time, gap
        int gap = 0;
        for (auto log : logs) {
            int i = 0;
            string id, time;
            bool start = false;
            while (log[i] != ':') id += log[i++];
            start = log[i + 1] == 's';
            while (log[++i] != ':');
            i++;
            while (i < log.size()) time += log[i++];
            int tid = stoi(id);
            int ttime = stoi(time);
            if (!start && !s.empty() && get<0>(s.top()) == tid) {
                int gap = get<2>(s.top());
                res[tid] += ttime + 1 - get<1>(s.top()) - gap;
                gap = ttime + 1 - get<1>(s.top());
                s.pop();
                if (!s.empty()) get<2>(s.top()) += gap;
            } else {
                s.push({tid, ttime, 0});
            }
        }
        return res;
    }
};
```



## 638. Shopping Offers

深搜，每次尝试选择一种优惠。直接用原函数递归好简洁。

```cpp
class Solution {
    bool check(vector<int>& needs, vector<int>& special) {
        for (int i = 0; i < needs.size(); i++) if (needs[i] < special[i]) return false;
        return true;
    }
public:
    int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs) {
        int res = 0;
        for (int i = 0; i < needs.size(); i++) res += needs[i]*price[i];
        for (int i = 0; i < special.size(); i++) {
            if (!check(needs, special[i])) continue;
            vector<int> next(needs);
            for (int k = 0; k < needs.size(); k++) next[k] -= special[i][k];
            res = min(res, special[i][needs.size()] + shoppingOffers(price, special, next));
        }
        return res;
    }
};
```

## 639. Decode Ways II

仔细考虑排列组合。

```cpp
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size(), d = 1e9+7;
        vector<long long> dp(n + 1, 0);
        dp[n] = 1;
        for (int i = n-1; i >= 0; i--) {
            dp[i] = s[i] == '*' ? 9*dp[i+1]%d : s[i] == '0' ? 0 : dp[i+1];
            if (i < n-1) {
                if (s[i] == '1') {
                    dp[i] += s[i+1] == '*' ? 9*dp[i+2]%d : dp[i+2];
                } else if (s[i] == '2') {
                    dp[i] += s[i+1] == '*' ? 6*dp[i+2]%d : s[i+1] <= '6' ? dp[i+2] : 0;
                } else if (s[i] == '*') {
                    dp[i] += s[i+1] == '*' ? 15*dp[i+2]%d : s[i+1] <= '6' ? 2*dp[i+2]%d : dp[i+2];
                }
            }
        }
        return dp[0] % d;
    }
};
```

## 641. Design Circular Deque

双向循环队列，用数组实现就行，很简单。

```cpp
class MyCircularDeque {
    int head;
    int tail;
    int size;
    int cap;
    vector<int> q;
public:
    /** Initialize your data structure here. Set the size of the deque to be k. */
    MyCircularDeque(int k): q(k, 0), head(0), tail(0), size(0), cap(k) {}
    
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    bool insertFront(int value) {
        if (size == cap) return false;
        if (--head < 0) head += cap;
        q[head] = value;
        size++;
        return true;
    }
    
    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    bool insertLast(int value) {
        if (size == cap) return false;
        size++;
        q[tail++] = value;
        if (tail >= cap) tail -= cap;
        return true;
    }
    
    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    bool deleteFront() {
        if (size == 0) return false;
        size--;
        if (++head >= cap) head -= cap;
        return true;
    }
    
    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    bool deleteLast() {
        if (size == 0) return false;
        size--;
        if (--tail < 0) tail += cap;
        return true;
    }
    
    /** Get the front item from the deque. */
    int getFront() {
        if (size == 0) return -1;
        else return q[head];
    }
    
    /** Get the last item from the deque. */
    int getRear() {
        if (size == 0) return -1;
        else return tail == 0 ? q.back() : q[tail-1];
    }
    
    /** Checks whether the circular deque is empty or not. */
    bool isEmpty() {
        return size == 0;
    }
    
    /** Checks whether the circular deque is full or not. */
    bool isFull() {
        return size == cap;
    }
};
```

## 646. Maximum Length of Pair Chain

首先我们可以按数对$(x,y)$的$x$排序，而后考虑$x$最大的元素$(x_{max},y)$，可以说明，以它为尾巴的链必然最长。这几乎是显然的，因为其他元素的$x$小于它，以其他元素为尾巴产生的限制越大。通过这个道理，我们只需贪心选择最后的元素即可。

```cpp
class Solution {
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end(), [](vector<int>& a, vector<int>& b) {
            return a[0] < b[0];
        });
        int left = INT_MAX, res = 0;
        for (auto i = pairs.rbegin(); i != pairs.rend(); i++) {
            if ((*i)[1] < left) {
                left = (*i)[0];
                res++;
            }
        }
        return res;
    }
};
```

## 647. Palindromic Substrings

简单的DP，也可以直接暴力地从每个点开始向两边扩展计数。

```cpp
class Solution {
public:
    int countSubstrings(string s) {
        int res = 0, n = s.size();
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (j - i <= 1) dp[i][j] = s[i] == s[j];
                else dp[i][j] = s[i] == s[j] && dp[i+1][j-1];
                res += dp[i][j];
            }
        }
        return res;
    }
};
```

## 648. Replace Words

把词根按首字母分类，按长度排序。寻找时找最短的匹配词根即可。也可以做一棵Trie，懒得写了。

```cpp
class Solution {
    bool check(string& t, string& a) {
        if (a.size() > t.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (a[i] != t[i]) return false;
        }
        return true;
    }
public:
    string replaceWords(vector<string>& dict, string sentence) {
        vector<vector<int>> m(26, vector<int>());
        for (int i = 0; i < dict.size(); i++) m[dict[i][0]-'a'].push_back(i);
        for (auto &v : m) {
            sort(v.begin(), v.end(), [&](int i, int j) {
                return dict[i].size() < dict[j].size();
            });
        }
        string res, tmp;
        sentence.push_back(' ');
        for (auto c : sentence) {
            if (c == ' ') {
                for (auto i : m[tmp[0]-'a']) {
                    if (check(tmp, dict[i])) {
                        res += dict[i] + " ";
                        tmp.clear();
                        break;
                    }
                }
                if (!tmp.empty()) {
                    res += tmp + " ";
                    tmp.clear();
                }
            } else tmp += c;
        }
        res.pop_back();
        return res;
    }
};
```



## 649. Dota2 Senate

通过计数简化操作，每次遇到R加一，遇到D减一。当负值时，如果当前是R，说明在这一轮他将被ban。

```cpp
class Solution {
public:
    string predictPartyVictory(string senate) {
        auto n = senate.size();
        int R = 0, D = 0, score = 0, first = 1;
        do {
            for (auto &c : senate) {
                if (c == 'R') {
                    R += first;
                    if (++score <= 0) {
                        c = '#';
                        R--;
                    }
                } else if (c == 'D') {
                    D += first;
                    if (--score >= 0) {
                        c = '#';
                        D--;
                    }
                }
            }
            first = 0;
        } while (R > 0 && D > 0);
        return R == 0 ? "Dire" : "Radiant";
    }
};
```

## 650. 2 Keys Keyboard

第一种解法：把它当成纯粹的DP来做，则需要$O(n^2)$的复杂度，具体思路是考虑剩余i个A，且剪贴板上的数量是k时的走向：一是复制，而是粘贴。递推公式不难想。

```cpp
class Solution {
public:
    int minSteps(int n) {
        vector<vector<int>> dp(n+1, vector<int>(n+1, 100000));
        for (int i = 0; i <= n; i++) {
            for (int k = n-i; k >= 0; k--) {
                if (i == 0) dp[i][k] = 0;
                else {
                    dp[i][k] = dp[i][n-i] + 1;
                    if (i >= k)
                        dp[i][k] = min(dp[i-k][k] + 1, dp[i][k]);
                }
            }
        }
        return dp[n-1][0];
    }
};
```

第二种解法：在我要生成n个A的时候，直观上，如果它是某个数A的倍数，则我只需要先生成A，再复制A数次即可，这样做必然比松散的随意复制来的快，以最小的成本达到了A。当然，如果有多种分解方法，则重复之，取最小值。这样从1开始，变成了一维的DP，但需要遍历i的所有可能的因子分解，复杂度为$O(n\sqrt n)$。

```cpp
class Solution {
public:
	int minSteps(int n) {
		vector<int> dp(n+1, 10000);
        dp[1] = dp[0] = 0;
        for (int i = 2; i <= n; i++) {
            for (int j = sqrt(i); j >= 1; j--) {
                if (i % j == 0) dp[i] = min(dp[i], min(dp[i/j]+j, dp[j]+i/j));
            }
        }
        return dp[n];
	}
};
```

第三种解法：从第二种解法中得到启发，我们实际上是不断地将n分解，假如分出了一个质数q乘合数m，则可以先生成m个，再重复q次更合算，否则可以把已生成的q个A看成一个整体A'，求解将A'重复m次的最小次数，显然这是一个进一步将m分解出质数的子问题。因此f(q*m)=f(m)+q，直到m不可分，则需要重复质数次。本问题转化为求所有质因子之和。我们从小质数开始不断地去除质因子即可。复杂度为$O(n)$。

```cpp
class Solution {
public:
	int minSteps(int n) {
		int res = 0, d = 2;
        while (n > 1) {
            while (n % d == 0) {
                res += d;
                n /= d;
            }
            d++;
        }
        return res;
	}
};
```



## 652. Find Duplicate Subtrees

重复出现的子树必然不会是自己的某个父节点，因此只有在子树全部访问完以后才加入map。通过根结点的值来构造map缩小搜索范围，逐一比较。任何重复的子树只会记录第一次出现的结点。粗略的估计复杂度为$O(n^2)$，但是由于map的优化以及不重复比较，实际运行效果还是很好的。

本题也可以将二叉树转化成字符串然后哈希，理论上复杂度降下来了。但是实际运行时间并不快多少，因为对字符串计算哈希值本身也是很大的开销。有个更tricky的哈希方法，不过个人感觉太tricky了，而且可能会翻车。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    bool equal(TreeNode* a, TreeNode* b) {
        if (!a && !b) return true;
        if (!a || !b) return false;
        return a->val == b->val && equal(a->left, b->left) && equal(a->right, b->right);
    }
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        unordered_map<int, vector<TreeNode*>> m;
        unordered_set<TreeNode*> u;
        function<void(TreeNode*)> trv = [&](TreeNode* t) {
            if (!t) return;
            bool isDup = false;
            for (auto prev : m[t->val]) {
                if (equal(prev, t)) {
                    if (u.find(prev) == u.end()) u.insert(prev);
                    isDup = true;
                    break;
                }
            }
            trv(t->left);
            trv(t->right);
            if (!isDup) m[t->val].push_back(t);
        };
        trv(root);
        return vector<TreeNode*>(u.begin(), u.end());
    }
};
```

## 654. Maximum Binary Tree

递归写法很简单。

```cpp
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        function<TreeNode*(int,int)> make = [&](int i, int j) -> TreeNode* {
            if (i == j) return new TreeNode(nums[i]);
            if (i > j) return nullptr;
            int k = i;
            for (int r = i; r <= j; r++) if (nums[r] > nums[k]) k = r;
            return new TreeNode(nums[k], make(i, k-1), make(k+1, j));
        };
        return make(0, nums.size() - 1);
    }
};
```

有一个巧妙的单调栈解法，可以$O(n)$。维护栈内元素的decreasing order，那么最近的比它大的元素就会是n的父亲，且是右孩子。中间pop出来的结点将被取代作为n的左孩子。

```cpp
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        vector<TreeNode*> s;
        for (auto n : nums) {
            auto cur = new TreeNode(n);
            while (!s.empty() && s.back()->val < n) {
                cur->left = s.back();
                s.pop_back();
            }
            if (!s.empty()) s.back()->right = cur;
            s.push_back(cur);
        }
        return s.front();
    }
};
```



## 658. Find K Closest Elements

二分查找到最近的点，然后向两边扩散。

```cpp
class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        auto right = lower_bound(arr.begin(), arr.end(), x);
        auto left = right;
        while (right - left < k) {
            if (right == arr.end()) {
                left = right - k;
            } else if (left == arr.begin()) {
                right = left + k;
            } else {
                if (abs(*prev(left) - x) <= abs(*right - x)) left--;
                else right++;
            }
        }
        return vector<int>(left, right);
    }
};
```

## 659. Split Array into Consecutive Subsequences

有点难想的贪心：使用两个哈希表，第一个维护剩余数量，第二个维护在n-1处结束的长度不小于3的子序列数目。每当新进来一个n，我们总是先想办法将其加在某条之前的子序列尾部，否则新建一条子序列并将下两个数字也加进去使其长度不小于3，如果找不到这样的下两个数字，那说明一定不能成功分隔。

```cpp
class Solution {
public:
    bool isPossible(vector<int>& nums) {
        unordered_map<int, int> count;
        unordered_map<int, int> lefts;
        for (auto n : nums) count[n]++;
        for (auto n : nums) {
            if (count[n] == 0) continue;
            if (lefts[n] != 0) {
                lefts[n]--;
                lefts[n+1]++;
            } else if (count[n+1] > 0 && count[n+2] > 0) {
                count[n+1]--;
                count[n+2]--;
                lefts[n+3]++;
            } else return false;
            count[n]--;
        }
        return true;
    }
};
```

## 662. Maximum Width of Binary Tree

为每层每个节点赋值，类似于前缀树，向左走则最低位补0，向右走补1，这样每层的宽度就是这个值的差值。遍历方式采用BFS的按层遍历。

```cpp
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        if (!root) return 0;
        int res = 1;
        deque<pair<TreeNode*, unsigned long long>> q;
        q.push_back({root, 0});
        while (!q.empty()) {
            int len = q.size();
            res = max(res, static_cast<int>(q.back().second - q.front().second + 1));
            for (int _ = 0; _ < len; _++) {
                auto t = q.front().first;
                auto v = q.front().second;
                if (t->left) q.push_back({t->left, v<<1});
                if (t->right) q.push_back({t->right, (v<<1)|1});
                q.pop_front();
            }
        }
        return res;
    }
};
```

## 664. Strange Printer

首先将连续的相同字符缩减为一个，减少问题规模。假设要打印`s[i:j]`的内容，尝试将其划分为两个子串，则划分后两个子串打印需要的turn数之和是足够的，那现在有可能`s[i]`与`s[j]`相等，此时我们可以考虑一次性把头尾都打印成`s[i]`，再进行覆盖，这样可以减少一次turn。就这样遍历所有的划分，求得其中的最小值即可。

```cpp
class Solution {
public:
    int strangePrinter(string s) {
        s.erase(unique(begin(s), end(s)), end(s));
        if (s.size() <= 2) return s.size();
        vector<vector<int>> dp(s.size(), vector<int>(s.size(), INT_MAX));
        for (int l = 0; l < s.size(); l++) {
            for (int i = 0; i+l < s.size(); i++) {
                if (l == 0) dp[i][i] = 1;
                else if (l == 1) dp[i][i+1] = 2;
                else for (int k = i; k < i+l; k++) {
                    dp[i][i+l] = min(dp[i][i+l],
                    dp[i][k]+dp[k+1][i+l]-static_cast<int>(s[i]==s[i+l]));
                }
            }
        }
        return dp[0][s.size()-1];
    }
};
```

## 665. Non-decreasing Array

注意细节。除了修改自身还可以修改前一个数字，并且后者优先考虑。

```cpp
class Solution {
public:
    bool checkPossibility(vector<int>& nums) {
        bool flag = false;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] < nums[i-1]) {
                if (flag) return false;
                if (i == 1) nums[i-1] = INT_MIN;
                else if (nums[i-2] <= nums[i]) nums[i-1] = nums[i-2];
                else nums[i] = nums[i-1];
                flag = true;
            }
        }
        return true;
    }
};
```

## 667. Beautiful Arrangement II

找规律题。

```cpp
class Solution {
public:
    vector<int> constructArray(int n, int k) {
        vector<int> res({1});
        res.reserve(n);
        int i = 2, j = n;
        bool pre = true;
        while (k-- > 1) {
            if (pre) res.push_back(j--);
            else res.push_back(i++);
            pre = !pre;
        }
        if (!pre) while (res.size() < n) res.push_back(j--);
        else while (res.size() < n) res.push_back(i++);
        return res;
    }
};
```



## 668. Kth Smallest Number in Multiplication Table

没有什么好的方法，就二分查找，注意lo和hi的计算很细致，因为mid不一定在矩阵中。

```cpp
class Solution {
public:
    int findKthNumber(int m, int n, int k) {
        int lo = 1, hi = m*n;
        while (lo < hi) {
            int mid = lo+(hi-lo)/2;
            int count = 0;
            for (int i = 1; i <= m; i++) {
                count += min(mid/i, n);
            }
            if (count >= k) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
};
```

## 669. Trim a Binary Search Tree

递归。

```cpp
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int L, int R) {
        if (!root) return nullptr;
        if (root->val < L)
            return trimBST(root->right, L, R);
        else if (root->val > R)
            return trimBST(root->left, L, R);
        else {
            root->left = trimBST(root->left, L, R);
            root->right = trimBST(root->right, L, R);
            return root;
        }
    }
};
```



## 673. Number of Longest Increasing Subsequence

最简单的DP解法，复杂度$O(n^2)$。

```cpp
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size(), maxL = 0, res = 0;
        vector<int> dp(n, 0);
        vector<int> count(n, 0);
        for (int i = 0; i < n; i++) {
            for (int j = i-1; j >= 0; j--) {
                if (nums[j] < nums[i] && dp[i] <= dp[j] + 1) {
                    if (dp[i] == dp[j] + 1) count[i] += count[j];
                    else {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    }
                }
            }
            if (dp[i] == 0) {
                dp[i] = 1;
                count[i] = 1;
            }
            maxL = max(maxL, dp[i]);
        }
        for (int i = 0; i < n; i++) {
            if (dp[i] == maxL)
                res += count[i];
        }
        return res;
    }
};
```



## 675. Cut Off Trees for Golf Event

最普通的BFS，写的时候要特别注意效率，写的差了本来能过得就超时了。

```cpp
class Solution {
    vector<pair<int, int>> dir = {{0,1},{1,0},{-1,0},{0,-1}};
    int bfs(int i, int j, int ri, int rj, vector<vector<int>>& f,
            int m, int n, vector<vector<int>>& v) {
        for (auto &row : v) for (auto &x : row) x = 0;
        v[i][j] = 1;
        queue<pair<int, int>> q({{i, j}});
        int dist = 0;
        while (!q.empty()) {
            int s = q.size();
            for (int _ = 0; _ < s; _++) {
                int i = q.front().first, j = q.front().second;
                q.pop();
                for (auto &[di, dj] : dir) {
                    int ni = i + di, nj = j + dj;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n && !v[ni][nj] && f[ni][nj]) {
                        if (ni == ri && nj == rj) return dist + 1;
                        v[ni][nj] = true;
                        q.push({ni, nj});
                    }
                }
            }
            dist++;
        }
        return -1;
    }
public:
    int cutOffTree(vector<vector<int>>& forest) {
        int m = forest.size(), n = forest[0].size();
        deque<pair<int, int>> trees;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++) {
                if (forest[i][j] > 1)
                    trees.push_back({forest[i][j], i*n+j});
            }
        }
        sort(trees.begin(), trees.end());
        vector<vector<int>> v(m, vector<int>(n, 0));
        int cur_i = 0, cur_j = 0, res = 0;
        if (trees.front().second == 0) trees.pop_front();
        for (auto tree : trees) {
            int i = tree.second/n, j = tree.second%n;
            int d = bfs(cur_i, cur_j, i, j, forest, m, n, v);
            if (d < 0) return -1;
            res += d;
            cur_i = i;
            cur_j = j;
        }
        return res;
    }
};
```

这是A*的解法，优化了200ms：

```cpp
class Solution {
    vector<pair<int, int>> dir = {{0,1},{1,0},{-1,0},{0,-1}};
    int bfs(int i, int j, int ri, int rj, vector<vector<int>>& f,
            int m, int n, vector<vector<int>>& d) {
        for (auto &row : d) for (auto &x : row) x = INT_MAX;
        d[i][j] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
        q.push({0, i*n+j});
        while (!q.empty()) {
            auto [_, pos] = q.top();
            int i = pos / n, j = pos % n;
            q.pop();
            if (d[i][j] < 0) continue;
            for (auto &[di, dj] : dir) {
                int ni = i + di, nj = j + dj;
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && f[ni][nj]) {
                    if (d[ni][nj] > d[i][j] + 1) {
                        d[ni][nj] = d[i][j] + 1;
                        if (ni == ri && nj == rj) return d[ni][nj];
                        q.push({abs(ni-ri)+abs(nj-rj)+d[ni][nj], ni*n+nj});
                    }
                }
            }
            d[i][j] = -1;
        }
        return -1;
    }
public:
    int cutOffTree(vector<vector<int>>& forest) {
        int m = forest.size(), n = forest[0].size();
        deque<pair<int, int>> trees;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++) {
                if (forest[i][j] > 1)
                    trees.push_back({forest[i][j], i*n+j});
            }
        }
        sort(trees.begin(), trees.end());
        vector<vector<int>> v(m, vector<int>(n, INT_MAX));
        int cur_i = 0, cur_j = 0, res = 0;
        if (trees.front().second == 0) trees.pop_front();
        for (auto tree : trees) {
            int i = tree.second/n, j = tree.second%n;
            int d = bfs(cur_i, cur_j, i, j, forest, m, n, v);
            if (d < 0) return -1;
            res += d;
            cur_i = i;
            cur_j = j;
        }
        return res;
    }
};
```

## 677. Map Sum Pairs

实现前缀树，并在结点处扩增额外信息，即该前缀下所有value的和，以遍快速查询。

```cpp
class MapSum {
    struct Node {
        int val;
        vector<Node*> kids;
        bool isKey;
        Node(): val(0), kids(26, nullptr), isKey(false) {}
        ~Node() {for (auto kid : kids) delete kid;}
    };
    Node root;
public:
    MapSum() {}
    
    void insert(string key, int val) {
        Node* cur = &root;
        vector<Node*> path({cur});
        for (char c : key) {
            if (!cur->kids[c-'a']) cur->kids[c-'a'] = new Node();
            cur = cur->kids[c-'a'];
            path.push_back(cur);
        }
        int diff = cur->isKey ? val - cur->val : val;
        cur->isKey = true;
        for (auto node : path) node->val += diff;
    }
    
    int sum(string prefix) {
        Node* cur = &root;
        for (auto c : prefix) {
            if (!cur->kids[c-'a']) return 0;
            cur = cur->kids[c-'a'];
        }
        return cur->val;
    }
};
```



## 679. 24 Game

straightforward dfs.

```cpp
class Solution {
public:
    bool judgePoint24(vector<int>& nums) {
        function<bool(vector<double>&)> dfs = [&](vector<double>& r) {
            if (r.size() == 1) return abs(r[0] - 24) < 1e-5;
            vector<double> tmp;
            tmp.resize(r.size() - 1);
            for (int i = 0; i < r.size(); i++) {
                for (int j = i+1; j < r.size(); j++) {
                    vector<double> t = {r[i]+r[j], r[i]*r[j], r[i]-r[j], r[j]-r[i]};
                    if (abs(r[i] - 0.0) > 1e-5) t.push_back(r[j]/r[i]);
                    if (abs(r[j] - 0.0) > 1e-5) t.push_back(r[i]/r[j]);
                    int index = 0;
                    for (auto k : t) {
                        tmp[index++] = k;
                        for (int l = 0; l < r.size(); l++) {
                            if (l!=i&&l!=j) tmp[index++] = r[l];
                        }
                        if (dfs(tmp)) return true;
                        index = 0;
                    }
                }
            }
            return false;
        };
        vector<double> t(nums.begin(), nums.end());
        return dfs(t);
    }
};
```



## 680. Valid Palindrome II

从两边开始向中间缩紧，遇到不同元素后尝试删除任意一个元素。

```cpp
class Solution {
public:
    bool validPalindrome(string s) {
        int i = 0, j = s.size() - 1, count = 0;
        while (i < j) {
            if (s[i] == s[j]) {
                i++; j--;
            } else {
                int i1 = i + 1, j1 = j;
                int i2 = i, j2 = j - 1;
                while (i1 < j1 && s[i1] == s[j1]) {
                    i1++; j1--;
                }
                while (i2 < j2 && s[i2] == s[j2]) {
                    i2++; j2--;
                }
                return i1 >= j1 || i2 >= j2;
            }
        }
        return true;
    }
};
```

## 684. Redundant Connection

无向图的情况下，可以通过深搜找环，但是使用并查集最简单。只要找从左到右第一个多余的边即可。

```cpp
class Solution {
    class DSU {
    private:
        vector<int> father;
        vector<int> height;
    public:
        DSU(int n): father(n, -1), height(n, 0) {}
        int find(int i) {
            if (father[i] == -1) return i;
            else return father[i] = find(father[i]);
        }
        bool uni(int i, int j) {
            int iroot = find(i);
            int jroot = find(j);
            if (iroot == jroot) return false;
            else if (height[iroot] > height[jroot]) {
                father[jroot] = iroot;
            } else if (height[iroot] < height[jroot]) {
                father[iroot] = jroot;
            } else {
                father[iroot] = jroot;
                height[jroot]++;
            }
            return true;
        }
    };
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        DSU u(edges.size() + 1);
        for (auto e : edges) {
            if (!u.uni(e[0], e[1])) return e;
        }
        return {-1, -1};
    }
};
```

## 685. Redundant Connection II

这题把上面一题变成了有向图，这样的话，多余的边有三种情况：

1. 图中没有入度为2的点，这说明原根结点在一个环里面，这种情况删除此环中任何一边都能成为树，找出这个环中最晚出现的边即可。
2. 图中有入度为2的结点，但没有成环，这说明某棵子树中的结点指向了另一棵子树，而非自己的祖先，这种情况删除此结点的任意一条入边即可，找最晚出现的那条入边。
3. 图中有入度为2的结点，且成环。那么一定是某结点指向了其祖先，这种情况下只能删除在环中且入度为2的那个结点。

整个过程通过DFS来实现，虽然有tricky的并查集实现：

```cpp
class Solution { 
public:
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<vector<int>> adj(n + 1, vector<int>());
        vector<int> count(n+1, 0);
        for (auto e : edges) {
            adj[e[0]].push_back(e[1]);
            count[e[1]]++;
        }
        vector<int> v(n + 1, 0);
        int candidate = -1;
        vector<int> res = {-1, -1};
        stack<int> s;
        unordered_set<int> cc;
        function<void(int)> dfs = [&](int i) {
            v[i] = 1;
            s.push(i);
            for (auto j : adj[i]) {
                if (v[j] == 2 && count[j] == 2) candidate = j;
                else if (v[j] == 1) {
                    int pre = j;
                    do {
                        if (count[pre] == 2) {
                            res = {s.top(), pre};
                            break;
                        }
                        cc.insert((s.top()<<16)|pre);
                        pre = s.top();
                        s.pop();
                    } while (pre != j);
                } else dfs(j);
            }
            if (!s.empty()) s.pop();
            v[i] = 2;
        };
        for (int i = 1; i <= n; i++) if (!v[i]) dfs(i);
        if (res[0] != -1) return res;
        if (candidate > 0) {
            for (int i = n-1; i >= 0; i--) {
                if (edges[i][1] == candidate) return edges[i];
            }
        } else if (!cc.empty()) {
            for (int i = n-1; i >= 0; i--) {
                if (cc.find((edges[i][0]<<16)|edges[i][1]) != cc.end()) return edges[i];
            }
        }
        return {-1, -1};// never happen here
    }
};
```

## 687. Longest Univalue Path

这题不难，但是得仔细考虑一下如何在一次遍历下求出解。办法是遍历中记录从左右孩子开始的等于当前节点值的最长路径`l`和`r`，这样`l+r`就是一条经过当前节点`t`的最长等值路径的长度，如此遍历即可。

```cpp
class Solution {
    int dfs(TreeNode* t, int f, int &res) {
        if (!t) return 0;
        int l = dfs(t->left, t->val, res);
        int r = dfs(t->right, t->val, res);
        res = max(res, l+r);
        if (t->val != f) return 0;
        else return max(l, r) + 1;
    }
public:
    int longestUnivaluePath(TreeNode* root) {
        int res = 0;
        dfs(root, INT_MAX, res);
        return res;
    }
};
```

## 688. Knight Probability in Chessboard

简单的DP，TopDown写法比较简单。

```cpp
class Solution {
public:
    double knightProbability(int N, int K, int r, int c) {
        unordered_map<int,double> mem;
        function<double(int,int,int)> dfs = [&](int i, int j, int k) {
            if (i < 0 || i >= N || j < 0 || j >= N) return 1.0;
            if (k == 0) return 0.0;
            int key = (i<<24)|(j<<16)|k;
            if (mem.find(key) != mem.end()) return mem[key];
            else return mem[key] = (
                dfs(i+2,j+1,k-1)+dfs(i+1,j+2,k-1)+
                dfs(i-2,j-1,k-1)+dfs(i-1,j-2,k-1)+
                dfs(i+2,j-1,k-1)+dfs(i-1,j+2,k-1)+
                dfs(i+1,j-2,k-1)+dfs(i-2,j+1,k-1)
            )/8.0;
        };
        return 1.0 - dfs(r, c, K);
    }
};
```



## 689. Maximum Sum of 3 Non-Overlapping Subarrays

设$dp[i][j]$为从$nums[j]$开始选取$i$个不重叠的子列所得的最大值。那么分两种情况，第一种是选择$nums[j:j+k]$这部分作为一个子列，剩余数组中取$i-1$个子列。另一种是不选择$nums[j]$开始的子列，那么自然就顺位到$dp[i][j+1]$了，最终取两者的最大值：
$$
dp[i][j]=max\{dp[i][j+1],\sum_{l=i}^{i+k-1}nums[l]+dp[i-1][j+k]\}
$$
由于第二部分的求和内容反复用到，因此先计算好放在`sum`数组中。这个问题还有一个麻烦是需要输出最优解，而不仅仅是最大值。因此用一个`trace`数组保存计算过程中每一步的选择（要么选`j`，要么不选，不选的话就往后找，选到了就将`i-1`），从而还原最优解。

```cpp
class Solution {
public:
    vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        vector<vector<int>> dp(3, vector<int>(n, 0));
        vector<vector<int>> trace(3, vector<int>(n, -1));
        vector<int> sum(n, 0);
        for (int j = 0; j <= n-k; j++) {
            if (j == 0) for (int l = 0; l < k; l++) sum[0] += nums[l];
            else sum[j] = sum[j-1] - nums[j-1] + nums[j+k-1];
        }
        int index = 0;
        int maxdp = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = n-(i+1)*k; j >= 0; j--) {
                if (i == 0) {
                    if (j+1 < n && dp[0][j+1] > sum[j]) dp[0][j] = dp[0][j+1];
                    else {
                        dp[0][j] = sum[j];
                        trace[0][j] = 1;
                    }
                } else {
                    if (dp[i][j+1] > sum[j] + dp[i-1][j+k]) dp[i][j] = dp[i][j+1];
                    else {
                        dp[i][j] = sum[j] + dp[i-1][j+k];
                        trace[i][j] = 1;
                    }
                    if (i == 2 && maxdp <= dp[i][j]) {
                        index = j;
                        maxdp = dp[i][j];
                    }
                }
            }
        }
        vector<int> res;
        int i = 2;
        while (i >= 0) {
            for (int j = index; j < n; j++) {
                if (trace[i][j] == 1) {
                    res.push_back(j);
                    index = j + k;
                    i--;
                    break;
                }
            }
        }
        return res;
    }
};
```

## 691. Stickers to Spell Word

DP，记录当前字符串对应的最小组合，通过深搜尝试选择sticker来构造新的字符串。一个优化：优先选择能够把首字母减掉的sticker，减小搜索空间——这个比较难想，并不会漏解，因为随着搜索的进行，首字母总是要被消除的。

```cpp
class Solution {
public:
    int minStickers(vector<string>& stickers, string target) {
        vector<vector<int>> count(stickers.size(), vector<int>(26, 0));
        for (int i = 0; i < stickers.size(); i++) {
            for (auto c : stickers[i]) {
                count[i][c-'a']++;
            }
        }
        unordered_map<string, int> dp;
        dp[""] = 0;
        function<int(string)> dfs = [&](string t) {
            if (dp.find(t) != dp.end()) return dp[t];
            vector<int> tmp(26, 0);
            for (auto x : t) tmp[x-'a']++;
            int res = INT_MAX;
            for (int i = 0; i < stickers.size(); i++) {
                if (count[i][t[0]-'a'] == 0) continue;
                string s;
                for (int j = 0; j < 26; j++) {
                    if (count[i][j] < tmp[j]) {
                        s += string(tmp[j] - count[i][j], char('a'+j));
                    }
                }
                if (s == t) continue;
                int ans = dfs(s);
                if (ans == -1) return -1;
                res = min(res, ans + 1);
            }
            if (res == INT_MAX) return -1;
            return dp[t] = res;
        };
        return dfs(target);
    }
};
```



## 692. Top K Frequent Words

用队列/红黑树存前k个。

```cpp
class Solution {
public:
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> m;
        for (auto &&w : words) m[w]++;
        auto cmp = [&](const string& a, const string& b) -> bool {return m[a] > m[b] || (m[a] == m[b] && a < b);};
        set<string, decltype(cmp)> s(cmp);
        for (auto &&[str, _] : m) {
            s.insert(str);
            if (s.size() > k) s.erase(prev(s.end()));
        }
        return vector<string>(s.begin(), s.end());
    }
};
```

## 695. Max Area of Island

深搜。

```cpp
class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        if (grid.empty() || grid[0].empty()) return 0;
        int res = 0, m = grid.size(), n = grid[0].size();
        function<int(int,int)> dfs = [&](int i, int j) {
            if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0) return 0;
            grid[i][j] = 0;
            return dfs(i+1, j) + dfs(i-1, j) + dfs(i, j+1) + dfs(i, j-1) + 1;
        };
        for (int i = 0; i < m ; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j]) res = max(res, dfs(i, j));
            }
        }
        return res;
    }
};
```

## 696. Count Binary Substrings

记录连续的0和1的数目。

```cpp
class Solution {
public:
    int countBinarySubstrings(string s) {
        int cur = 1, pre = 0, res = 0;
        for (int i = 1; i < s.size(); i++) {
            if (s[i] != s[i-1]) {
                pre = cur;
                cur = 0;
            }
            if (++cur <= pre) res++;
        }
        return res;
    }
};
```

## 698. Partition to K Equal Sum Subsets

可以简单的用哈希表来存mask，避免重复情况。下面是更聪明的避免重复地搜索方法，用一个idx限制下一个添加的元素的位置，不重复地添加一组元素直到和达到目标值，此时将idx清零，组合下一组数，直到k组数都组合完成。

```cpp
class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = 0, len = nums.size();
        for (auto n : nums) sum += n;
        if (sum % k != 0) return false;
        sum /= k;
        sort(nums.rbegin(), nums.rend());
        function<bool(int,int,int,int)> dfs = [&](int target, int mask, int step, int idx) {
            if (target == 0) {
                if (++step == k) return true;
                return dfs(sum, mask, step, 0);
            }
            for (int i = idx; i < len; i++) {
                if (!(mask&(1<<i)) && nums[i] <= target) {
                    if (dfs(target - nums[i], mask|(1<<i), step, i+1)) return true;
                }
            }
            return false;
        };
        return dfs(sum, 0, 0, 0);
    }
};
```

## 713. Subarray Product Less Than K

滑动窗口题。考虑以r结尾的连乘，如果以r-1的连乘最多乘到l，则r时，无法乘到l之前。整个循环l和r递增，$O(n)$。

```cpp
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int l = 0, count = 0;
        int64_t prod = 1;
        if (k <= 1) return 0;
        for (int r = 0; r < nums.size(); r++) {
            prod *= nums[r];
            while (prod >= k) prod /= nums[l++];
            count += r - l + 1;
        }
        return count;
    }
};
```

## 726. Number of Atoms

用栈保存计数，遇到右括号则将最靠近的左括号间的数量翻倍。

```cpp
class Solution {
public:
    string countOfAtoms(string f) {
        list<pair<string, int>> s;
        int i = 0;
        string tmp, count;
        while (i < f.size()) {
            if (isupper(f[i])) {
                tmp += f[i++];
                while (islower(f[i])) {
                    tmp += f[i++];
                }
                if (i == f.size() || !isdigit(f[i])) {
                    s.push_back({tmp, 1});
                    tmp.clear();
                }
            } else if (isdigit(f[i])) {
                while (isdigit(f[i])) {
                    count += f[i++];
                }
                s.push_back({tmp, stoi(count)});
                tmp.clear();
                count.clear();
            } else if (f[i] == '(') {
                s.push_back({"#", 1});
                i++;
            } else {
                i++;
                while (isdigit(f[i])) {
                    count += f[i++];
                }
                int t = stoi(count);
                count.clear();
                auto p = --s.end();
                while (p->first != "#") {
                    p->second *= t;
                    p--;
                }
                s.erase(p);
            }
        }
        map<string, int> m;
        for (auto &&p : s) {
            m[p.first] += p.second;
        }
        string res;
        for (auto& p : m) {
            res += p.first;
            if (p.second > 1)
                res += to_string(p.second);
        }
        return res;
    }
};
```

## 730. Count Different Palindromic Subsequences

注意必须是不重复的回文串，因此需要在abcd这个限制上做文章；构成目标回文串的组合必然是以aa，bb，cc，dd包裹的，否则便是单独的a，b，c，d。因此我们分别记录，另`dp[i][j][x]`为`[i:j+1]`中单独的`x`或者`xx`包裹的非空不重复回文串数量；然后根据`s[i]`和`s[j]`来缩小范围，确定递推式即可。

```cpp
class Solution {
public:
    int countPalindromicSubsequences(string S) {
        int64_t mod = 1e9+7;
        int n = S.size();
        int64_t dp[1000][1000][4] = {0};
        for (int i = n-1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                for (int x = 0; x < 4; x++) {
                    char c = 'a' + x;
                    if (i == j) {
                        dp[i][j][x] = S[i] == c ? 1 : 0;
                    } else if (S[i] == c && S[j] == c) {
                        dp[i][j][x] = 2 + dp[i+1][j-1][0] + dp[i+1][j-1][1]
                                        + dp[i+1][j-1][2] + dp[i+1][j-1][3];
                    } else if (S[i] != c) {
                        dp[i][j][x] = dp[i+1][j][x];
                    } else {
                        dp[i][j][x] = dp[i][j-1][x];
                    }
                    dp[i][j][x] %= mod;
                }
            }
        }
        return (dp[0][n-1][0] + dp[0][n-1][1] + dp[0][n-1][2] + dp[0][n-1][3]) % mod;
    }
};
```

## 740. Delete and Earn

注意是把相邻数字全部删去，因此问题简单了。

```cpp
class Solution {
public:
    int deleteAndEarn(vector<int>& nums) {
        int count[10001] = {0};
        for (auto n : nums) count[n]++;
        int x = 0, y = 0, t;
        for (int i = 1; i < 10001; i++) {
            t = max(y, x + i*count[i]);
            x = y;
            y = t;
        }
        return y;
    }
};
```

## 743. Network Delay Time

Dijkstra；

```cpp
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int N, int K) {
        vector<vector<pair<int, int>>> adj(N+1);
        vector<int> d(N+1, INT_MAX);
        vector<int> v(N+1, 0);
        for (const auto &v : times) {
            adj[v[0]].push_back(make_pair(v[1], v[2]));
        }
        int finished = 0, res = 0;
        d[K] = 0;
        auto cmp = [](const auto &a, const auto &b) {
            return a.second > b.second;
        };
        priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> q(cmp);
        q.push(make_pair(K, d[K]));
        while (!q.empty()) {
            auto t = q.top();
            int i = t.first;
            q.pop();
            if (v[i] == 1) continue;
            v[i] = 1;
            finished++;
            res = max(res, d[i]);
            for (auto& next : adj[i]) {
                if (v[next.first] == 0 && d[next.first] > d[i] + next.second) {
                    d[next.first] = d[i] + next.second;
                    q.push(make_pair(next.first, d[next.first]));
                }
            }
        }
        return finished == N ? res : -1;
    }
};
```

## 752. Open the Lock

广搜，一开始想岔了。

```cpp
class Solution {
public:
    int openLock(vector<string>& deadends, string target) {
        int dp[10000], T = stoi(target);
        int div[4] = {1, 10, 100, 1000};
        fill(dp, dp+10000, INT_MAX);
        for (const auto& s : deadends) {
            dp[stoi(s)] = -1;
        }
        if (dp[0] == -1) return -1;
        dp[0] = 0;
        queue<int> q({0});
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            if (cur == T) break;
            for (int d : div) {
                int bit = (cur/d)%10;
                if (bit <= 8 && dp[cur+d] == INT_MAX) {
                    dp[cur+d] = dp[cur] + 1;
                    q.push(cur+d);
                }
                if (bit >= 1 && dp[cur-d] == INT_MAX) {
                    dp[cur-d] = dp[cur] + 1;
                    q.push(cur-d);
                }
                if (bit == 0 && dp[cur+9*d] == INT_MAX) {
                    dp[cur+9*d] = dp[cur] + 1;
                    q.push(cur+9*d);
                }
                if (bit == 9 && dp[cur-9*d] == INT_MAX) {
                    dp[cur-9*d] = dp[cur] + 1;
                    q.push(cur-9*d);
                }
            }
        }
        return dp[T] == INT_MAX ? -1 : dp[T];
    }
};
```

## 754. Reach the Number

求和，如果与目标的差是一个偶数，则总是可以找到前面的一些数使得改成负号可以达到目标，否则不行。

```cpp
class Solution {
public:
    int reachNumber(int target) {
        target = abs(target);
        int i = 1, s = 0;
        while (true) {
            s += i;
            if (s >= target && (s-target) % 2 == 0) {
                return i;
            }
            i++;
        }
        return -1;
    }
};
```

## 756. Pyramid Transition Matrix

深搜，递归稍微复杂了点，容易出错。

```cpp
class Solution {
    bool solve(string& b, unordered_map<int, string>& m, 
               int i, string& tmp) {
        if (b.size() == 1) return true;
        if (i == b.size() - 1) {
            string t;
            return solve(tmp, m, 0, t);
        }
        int k = (b[i]-'A')*10+b[i+1]-'A';
        if (m.find(k) == m.end()) return false;
        for (auto c : m[k]) {
            tmp.push_back(c);
            if (solve(b, m, i+1, tmp))
                return true;
            tmp.pop_back();
        }
        return false;
    }
public:
    bool pyramidTransition(string bottom, vector<string>& allowed) {
        unordered_map<int, string> m;
        for (const auto& s : allowed) {
            m[(s[0]-'A')*10+s[1]-'A'].push_back(s[2]);
        }
        string t;
        return solve(bottom, m, 0, t);
    }
};
```

## 757. Set Intersection Size At Least Two

这题是之前452题戳气球的进阶版，相当于每个区间需要戳至少两次。思路类似。

按照区间尾排序，贪心，尝试选择区间尾的两个数，如果当前区间不包含之前选择的两个数，则需要新选两个数；如果只包含一个，那么小的那个必然不包含，新选择一个应当是新区间的尾v[1]，这里需要考虑一个特殊情况——b可能恰好是区间尾，此时新选择的应该是v[1]-1。

```cpp
class Solution {
public:
    int intersectionSizeTwo(vector<vector<int>>& intervals) {
        sort(begin(intervals), end(intervals), [](const vector<int>& a, const vector<int>& b) {
            return a[1] < b[1];
        });
        int a = -1, b = -1, res = 0;
        for (const auto& v : intervals) {
            if (v[0] > b) {
                res += 2;
                a = v[1]-1;
                b = v[1];
            } else if (a < v[0]) {
                res++;
                a = b == v[1] ? v[1] - 1 : b;
                b = v[1];
            }
        }
        return res;
    }
};
```

## 761. Special Binary String

难想，把父串分成不可分的special子串，递归处理子special串，子串去头尾的1和0还是special串（否则就可分），因此对不可分的串递归求解其最大的变化，最后把所有不可分串排序即可。

```cpp
class Solution {
public:
    string makeLargestSpecial(string S) {
        function<string(int,int)> dfs = [&](int s, int t) {
            int count = 0, i = s, j = s;
            vector<string> slices;
            while (j < t) {
                if (S[j] == '1') count++;
                else count--;
                if (count == 0) {
                    slices.push_back("1"+dfs(i+1, j)+"0");
                    i = j+1;
                }
                j++;
            }
            sort(rbegin(slices), rend(slices));
            string r;
            for (auto&& sub : slices) r += sub;
            return r;
        };
        return dfs(0, S.size());
    }
};
```

## 763. Partition Labels

实际上就是计算区间族中合并相交区间后剩余的区间数。

```cpp
class Solution {
public:
    vector<int> partitionLabels(string S) {
        vector<pair<int, int>> v(26, make_pair(-1, -1));
        for (int i = 0; i < S.size(); i++) {
            if (v[S[i]-'a'].first == -1) {
                v[S[i]-'a'] = make_pair(i, i);
            } else {
                v[S[i]-'a'].second = i;
            }
        }
        sort(begin(v), end(v));
        vector<int> res;
        int left = -1, right = -1;
        for (const auto& p : v) {
            if (p.first == -1) continue;
            if (left == -1) {
                left = p.first;
                right = p.second;
            } else if (p.first > right) {
                res.push_back(right - left + 1);
                left = p.first;
                right = p.second;
            } else {
                right = max(right, p.second);
            }
        }
        res.push_back(right - left + 1);
        return res;
    }
};
```

## 767. Reorganize String

如果有一个字母出现超过一半，就不可以，否则我们只要按出现次数从大到小排序，然后循环填奇偶位置即可。

```cpp
class Solution {
public:
    string reorganizeString(string S) {
        int c[26] = {0}, n = S.size();
        for (auto x : S) c[x-'a']++;
        for (auto i : c) if (i > (n+1)/2) return "";
        string res(n, ' ');
        sort(begin(S), end(S), [&](const auto& a, const auto& b) {
            return c[a-'a'] > c[b-'a'] || (c[a-'a'] == c[b-'a'] && a < b);
        });
        int i = 0, j = 0;
        while (j < n) {
            if (i >= n) i = 1;
            res[i] = S[j++];
            i += 2;
        }
        return res;
    }
};
```

## 769. Max Chunks To Make Sorted

非常简单，看透了即可。

```cpp
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int m = -1, res = 0;
        for (int i = 0; i < arr.size(); i++) {
            m = max(m, arr[i]);
            if (m <= i) res++;
        }
        return res;
    }
};
```

## 768. Max Chunks To Make Sorted II

不懂这题为啥Hard，明明很简单。

```cpp
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int n = arr.size(), m = INT_MIN, res = 0;
        vector<int> mins(n+1, INT_MAX);
        for (int i = n-1; i >= 0; --i) {
            mins[i] = min(arr[i], mins[i+1]);
        }
        for (int i = 0; i < n; i++) {
            m = max(m, arr[i]);
            if (m <= mins[i+1]) res++;
        }
        return res;
    }
};
```

## 773. Sliding Puzzle

普通的广搜，用int来存状态，也可以直接做成string。

```cpp
class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        auto state = [&]() -> int {
            return 0 | (board[0][0] << 25) | (board[0][1] << 20) 
                | (board[0][2] << 15) | (board[1][0] << 10) | (board[1][1] << 5) | board[1][2];
        };
        auto restore = [&](int k) {
            board[0][0] = k >> 25;
            board[0][1] = (k>>25<<5)^(k>>20);
            board[0][2] = (k>>20<<5)^(k>>15);
            board[1][0] = (k>>15<<5)^(k>>10);
            board[1][1] = (k>>10<<5)^(k>>5);
            board[1][2] = (k>>5<<5)^k;
        };
        vector<vector<int>> dr({{0,1},{0,-1},{1,0},{-1,0}});
        int target = 0|(1<<25)|(2<<20)|(3<<15)|(4<<10)|(5<<5)|0;
        unordered_map<int, int> v;
        v[state()] = 0;
        queue<int> q({state()});
        while (!q.empty()) {
            int t = q.front();
            q.pop();
            if (t == target) return v[t];
            restore(t);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    if (board[i][j] == 0) {
                        for (const auto& d : dr) {
                            int ni = i + d[0], nj = j + d[1];
                            if (ni >= 0 && ni < 2 && nj >= 0 && nj < 3) {
                                swap(board[i][j], board[ni][nj]);
                                int k = state();
                                if (v.find(k) == v.end()) {
                                    v[k] = v[t] + 1;
                                    q.push(k);
                                }
                                swap(board[i][j], board[ni][nj]);
                            }
                        }
                        break;
                    }
                }
            }
        }
        return -1;
    }
};
```

## 775. Global and Loacal Inversions

比较隔一位的最小值即可。

```cpp
class Solution {
public:
    bool isIdealPermutation(vector<int>& A) {
        int n = A.size(), m = INT_MIN;
        for (int i = 1; i < n; i++) {
            if (A[i] < m) return false;
            m = max(m, A[i-1]);
        }
        return true;
    }
};
```

## 779. K-th Symbol in Grammar

有趣的递归。

```cpp
class Solution {
public:
    int kthGrammar(int N, int K) {
        // f n k depends on f n-1 k/2
        if (N == 1) return 0;
        if (kthGrammar(N-1, (K+1)/2) == 0) {
            return K&1 ? 0 : 1;
        } else {
            return K&1 ? 1 : 0;
        }
    }
};
```



## 793. Preimage Size of Factorial Zeroes Function

一个阶乘后面0的个数实际上等于乘积中因子10的数量，而2的数量显然是大于5的，因此每当遇到5的倍数，0的数量会增加。因此一旦存在一个x，使得f(x)=K，则一定有5个x满足条件（从某个5的倍数开始的5个数），而如果我们找不到x，则不存在。因此直接二分查找这样的x即可，如果找到就返回5，不然就返回0。查找的上界是5*K，因为这个数的阶乘后面至少跟K个零。

```cpp
class Solution {
public:
    int preimageSizeFZF(int K) {
        auto f = [](int64_t n) {
            n -= n % 5;
            int r = 0;
            while (n >= 5) {
                n /= 5;
                r += n;
            }
            return r;
        };
        int64_t lo = 0, hi = 5*(int64_t)K+1;
        while (hi-lo > 1) {
            int64_t mid = lo + (hi - lo)/2;
            if (f(mid) > K) hi = mid;
            else lo = mid;
        }
        if (f(lo) != K) return 0;
        else return 5;
    }
};
```

## 801. Minimum Swaps To Make Sequences Increasing

DP，记录0至i-1位替换/不替换所需要的最小替换次数。

```cpp
class Solution {
public:
    int minSwap(vector<int>& A, vector<int>& B) {
        int n = A.size();
        vector<int> dp0(n, 1000);
        vector<int> dp1(n, 1000);
        dp1[0] = 1;
        dp0[0] = 0;
        for (int i = 1; i < n; i++) {
            if (A[i] > A[i-1] && B[i] > B[i-1]) {
                dp0[i] = dp0[i-1];
                dp1[i] = dp1[i-1] + 1;
            }
            if (A[i] > B[i-1] && B[i] > A[i-1]) {
                dp0[i] = min(dp0[i], dp1[i-1]);
                dp1[i] = min(dp1[i], dp0[i-1] + 1);
            }
        }
        return min(dp0[n-1], dp1[n-1]);
    }
};
```

## 802. Find Eventual Safe States

如果结点i的所有后继结点都是safe的，那么结点i是safe的，否则不是，因此DFS解决。

```cpp
class Solution {
public:
    vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        int n = graph.size();
        vector<int> v(n, -1);
        vector<int> res;
        function<bool(int)> dfs = [&](int i) {
            if (v[i] >= 0) return v[i] == 1;
            if (graph[i].empty()) {
                v[i] = 1;
            } else {
                v[i] = 0;
                for (auto j : graph[i]) {
                    if (!dfs(j)) {
                        v[i] = 2;
                        return false;
                    }
                }
                v[i] = 1;
            }
            return v[i] == 1;
        };
        for (int i = 0; i < n; i++) if (dfs(i)) res.push_back(i);
        sort(res.begin(), res.end());
        return res;
    }
};
```

## 803. Bricks Falling When Hit

反向计算，当每次把砖头放回去的时候，会增加多少not Fall的砖块，即增加了多少连通到顶部的砖块。一开始使用并查集记录各连通集的大小和内容，然后使用notFall数组记录not fall的连通集id。最后反向遍历hits。

```cpp
class Solution {
    class UnionFindSet {
        vector<int> father;
        vector<int> rank;
        vector<int> size;
    public:
        UnionFindSet(int s): father(s, 0), rank(s, 0), size(s, 0) {
            for (int i = 0; i < s; i++) {
                father[i] = i;
                size[i] = 1;
            }
        }
        int Find(int x) {
            if (x != father[x]) {
                father[x] = Find(father[x]);
            }
            return father[x];
        }
        void Union(int x, int y) {
            int xroot = Find(x);
            int yroot = Find(y);
            if (xroot == yroot) return;
            if (rank[xroot] < rank[yroot]) {
                father[xroot] = yroot;
                size[yroot] += size[xroot];
            }
            else if (rank[xroot] > rank[yroot]) {
                father[yroot] = xroot;
                size[xroot] += size[yroot];
            }
            else {
                father[xroot] = yroot;
                size[yroot] += size[xroot];
                rank[yroot]++;
            }
        }
        int getSize(int x) {
            return size[Find(x)];
        }
    };
public:
    vector<int> hitBricks(vector<vector<int>>& grid, vector<vector<int>>& hits) {
        for (const auto &p : hits) if(grid[p[0]][p[1]] == 1) grid[p[0]][p[1]] = 2;
        int m = grid.size(), n = grid[0].size();
        UnionFindSet u(m*n);
        vector<vector<int>> dr({{1,0},{0,1},{-1,0},{0,-1}});
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    for (const auto &d : dr) {
                        int ni = i+d[0], nj = j+d[1];
                        if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == 1)
                            u.Union(i*n+j, ni*n+nj);
                    }
                }
            }
        }
        vector<bool> notFalls(m*n, false);
        for (int j = 0; j < n; j++) {
            if (grid[0][j] == 1) notFalls[u.Find(j)] = true;
        }
        vector<int> res;
        for (auto it = hits.rbegin(); it != hits.rend(); it++) {
            const auto &hit = *it;
            if (grid[hit[0]][hit[1]] == 0) {
                res.push_back(0);
                continue;
            }
            grid[hit[0]][hit[1]] = 1;
            int count = 0;
            bool up = hit[0] == 0;
            int pos = hit[0]*n+hit[1];
            for (const auto &d : dr) {
                int ni = hit[0]+d[0], nj = hit[1]+d[1];
                int nextPos = ni*n+nj;
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == 1 
                    && u.Find(pos) != u.Find(nextPos)) {
                    if (notFalls[u.Find(nextPos)]) {
                        up = true;
                        u.Union(pos, nextPos);
                        notFalls[u.Find(pos)] = true;
                    } else {
                        count += u.getSize(nextPos);
                        u.Union(pos, nextPos);
                    }
                }
            }
            if (up) {
                res.push_back(count);
                notFalls[u.Find(pos)] = true;
            } else {
                res.push_back(0); 
            }
        }
        reverse(begin(res), end(res));
        return res;
    }
};
```

## 818. Race Car

按照与目标点的距离DP，先贪心地尽可能接近目标点，走k步，其中距离d应当在$(2^{k-1}-1,2^{k}-1]$，此时若能够恰好到达，则显然最短步数是k步，若不能，有两种情况：

1. 走k步，然后用R转弯，新的距离为$2^k-1-d$。
2. 走k-1步，然后倒退m<k-1步，再把方向转回来。注意这里可以倒退零步，这样的效果是速度重置为1。

```cpp
class Solution {
public:
    int racecar(int target) {
        vector<int> dp(target+1, -1);
        function<int(int)> dfs = [&](int d) {
            if (dp[d] >= 0) return dp[d];
            int k = 1;
            while ((1<<k)-1 < d) k++;
            if ((1<<k) - 1 == d) return dp[d] = k;
            int step = (1<<k) - 1;
            dp[d] = dfs(step-d) + 1 + k;
            int t = d;
            t -= (1<<(k-1)) - 1;
            for (int m = 0; m < k-1; m++) {
                dp[d] = min(dp[d], dfs(t+(1<<m)-1)+k+m+1);
            }
            return dp[d];
        };
        return dfs(target);
    }
};
```

## 827. Making A Large Island [DFS]

首先通过深搜确定现存的岛屿以及其面积（保存最大值），用数组v保存每个格子属于哪个岛。然后遍历所有0的格子，判断其与哪些相邻岛屿相连，而后添加这些面积，再次更新最大值。复杂度$O(mn)$。

```cpp
class Solution {
public:
    int largestIsland(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), id = 0;
        int dx[] = {1,-1,0,0};
        int dy[] = {0,0,1,-1};
        vector<vector<int>> v(m, vector<int>(n, -1));
        function<void(int,int,int,int&)> dfs = [&](int i, int j, int d, int& s) {
            v[i][j] = d;
            s++;
            for (int k = 0; k < 4; k++) {
                int ni = i + dx[k];
                int nj = j + dy[k];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == 1 && v[ni][nj] == -1)
                    dfs(ni, nj, d, s);
            }
        };
        vector<int> areas;
        int maxArea = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && v[i][j] == -1) {
                    int s = 0;
                    dfs(i, j, id++, s);
                    areas.push_back(s);
                    maxArea = max(maxArea, s);
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    int a = 1;
                    vector<int> u;
                    for (int k = 0; k < 4; k++) {
                        int ni = i + dx[k];
                        int nj = j + dy[k];
                        if (ni >= 0 && ni < m && nj >= 0 && nj < n &&
                            grid[ni][nj] == 1 && find(begin(u), end(u), v[ni][nj]) == end(u)) {
                            a += areas[v[ni][nj]];
                            u.push_back(v[ni][nj]);
                        }
                    }
                    maxArea = max(maxArea, a);
                }
            }
        }
        return maxArea;
    }
};
```

## 834. Sum of Distances in Tree

这题需要动点心思，进行两次相同顺序的深搜，第一次我们计算每个节点的子孙数量以及到子孙的距离和，分别为pair的first和second，这很好算。显然，第一次遍历完成后，根结点0到所有子孙的距离和就是我们需要的结果了，但是其他结点还需进一步计算。

因此在第二次遍历的时候，我们记录当前结点的父亲结点，而后假设父亲结点到所有其他结点的距离和已经计算好了，那么当前结点到其他结点的距离可以通过父亲结点及之前计算的到子孙的距离和来得出：

`res[i] = info[i].second + res[f] - (info[i].first + info[i].second + 1) + N - info[i].first - 1;`

化简一下就是下面程序中的式子。仔细思考一下就能理解的，懒得写清楚解释了。

```cpp
class Solution {
public:
    vector<int> sumOfDistancesInTree(int N, vector<vector<int>>& edges) {
        vector<int> res(N, -1);
        vector<pair<int, int>> info(N, make_pair(-1, 0));
        vector<vector<int>> adj(N);
        for (auto e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        function<void(int)> dfs_0 = [&](int i) {
            info[i].first = 0;
            for (auto j : adj[i]) {
                if (info[j].first < 0) {
                    dfs_0(j);
                    info[i].first += info[j].first+1;
                    info[i].second += info[j].first+1+info[j].second;
                }
            }
        };
        dfs_0(0);
        function<void(int,int)> dfs_1 = [&](int i, int f) {
            if (f < 0) res[i] = info[i].second;
            else res[i] = res[f] + N - 2*info[i].first - 2;
            for (auto j : adj[i]) {
                if (res[j] < 0) dfs_1(j, i);
            }
        };
        dfs_1(0, -1);
        return res;
    }
};
```

## 870. Advantage Shuffle

对A排序，贪心选择A中最小的能够大于B的元素。需要记录ID，可以用map或者做成pair对再排序。

```cpp
class Solution {
public:
    vector<int> advantageCount(vector<int>& A, vector<int>& B) {
        vector<int> res(B.size(), -1), tmp;
        sort(begin(A), end(A));
        multimap<int, int> m;
        for (int i = 0; i < B.size(); i++) m.insert({B[i], i});
        int i = 0;
        for (auto &[n, id] : m) {
            while (i < B.size() && A[i] <= n) {
                tmp.push_back(A[i++]);
            }
            if (i == B.size()) break;
            res[id] = A[i++];
        }
        i = 0;
        for (auto& n : res) if (n < 0) n = tmp[i++];
        return res;
    }
};
```

## 878. Nth Magical Number

考虑12和15，将其除以最大公约数3，得到两个互质的整数4和5，那么我们只要考虑4和5组成的第N大魔术数，然后乘以3即可。

列出来如下：4,5,8,10,12,15,16,20,24,25,28,30.....可以看到这个序列模4和5的最小公倍数20是循环的。实际上很直观，因此我们只需计算第N个数在第几次循环，就可以按照这个性质计算出具体的值了。

如何计算循环的周期，还是看4和5，4的倍数为4,8,12,16,20，5的倍数为5,10,15,20，实际上是4+5-1。这些都很好理解。因此问题就简单了，先找出循环数，然后计算偏移量相加即可。

```cpp
class Solution {
    int gcd(int a, int b) {
        if (a == 0) return b;
        return gcd(b%a, a);
    }
public:
    int nthMagicalNumber(int N, int A, int B) {
        int M = 1e9+7;
        int g = gcd(min(A, B), max(A, B));
        A /= g; B /= g;
        int count = A+B-1;
        int n = (N-1)/count, off = (N-1)%count;
        int64_t res = static_cast<int64_t>(A*B)*n;
        int mA = A, mB = B, t = 0;
        for (int i = 0; i < off+1; i++) {
            if (mA <= mB) {
                t = mA;
                mA += A;
            } else {
                t = mB;
                mB += B;
            }
        }
        res += t;
        return (res*g)%M;
    }
};
```

## 891. All Possible FBT

递归生成，利用对称性尽量减少递归调用次数。

```cpp
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
 */
class Solution {
public:
    vector<TreeNode*> allPossibleFBT(int N) {
        if (N%2 == 0) return {};
        if (N == 1) return {new TreeNode(0)};
        vector<TreeNode*> res;
        for (int L = 1; L <= N/2; L += 2) {
            auto left = allPossibleFBT(L);
            auto right = L == N-1-L ? left : allPossibleFBT(N-1-L);
            for (auto l : left) {
                for (auto r : right) {
                    res.push_back(new TreeNode(0, l, r));
                    if (L != N-1-L) res.push_back(new TreeNode(0, r, l));
                }
            }
        }
        return res;
    }
};
```

## 913. Cat and Mouse

```cpp
class Solution {
public:
    int catMouseGame(vector<vector<int>>& graph) {
        int n = graph.size();
        int state[50][50][2], count[50][50][2];
        bool visited[50][50][2];
        for (int m = 0; m < n; m++) {
            for (int c = 0; c < n; c++) {
                count[m][c][0] = graph[m].size();
                count[m][c][1] = graph[c].size();
                for (int t = 0; t <= 1; t++) {
                    state[m][c][t] = 0;// draw
                    visited[m][c][t] = false;
                    if (m == 0 && c > 0) state[m][c][t] = 1;// mouth wins
                    if (m > 0 && m == c) state[m][c][t] = 2;// cat wins
                }
            }
        }
        for (int m = 0; m < n; m++) {
            for (int c : graph[0]) {
                count[m][c][1]--;
            }
        }
        auto adjs = [&](int m, int c, int t) {
            vector<tuple<int,int,int>> r;
            if (t == 0) {
                for (auto cc : graph[c]) {
                    if (cc != 0) {
                        r.push_back({m, cc, 1-t});
                    }
                }
            } else {
                for (auto mm : graph[m]) {
                    r.push_back({mm, c, 1-t});
                }
            }
            return r;
        };
        function<void(int,int,int)> dfs = [&](int m, int c, int t) {
            if (visited[m][c][t]) return;
            visited[m][c][t] = true;
            for (auto [mm, cc, tt] : adjs(m, c, t)) {
                if (state[mm][cc][tt] != 0) continue;
                if (tt == 0 && state[m][c][t] == 1 ||
                    tt == 1 && state[m][c][t] == 2) {
                    state[mm][cc][tt] = tt == 0 ? 1 : 2;
                    dfs(mm, cc, tt);
                } else {
                    if (--count[mm][cc][tt] == 0) {
                        state[mm][cc][tt] = tt == 0 ? 2 : 1;
                        dfs(mm, cc, tt);
                    }
                }
            }
        };
        for (int m = 0; m < n; m++) {
            for (int c = 0; c < n; c++) {
                for (int t = 0; t < 2; t++) {
                    if (state[m][c][t] != 0) dfs(m, c, t);
                }
            }
        }
        return state[1][2][0];
    }
};
```

## 939. Minimum Area Rectangle

按照列按序保存每列的纵坐标值，然后遍历所有列对，寻找相同的纵坐标对，计算面积。

```cpp
class Solution {
public:
    int minAreaRect(vector<vector<int>>& points) {
        unordered_map<int, set<int>> m;
        for (auto &p : points) {
            m[p[0]].insert(p[1]);
        }
        int res = INT_MAX;
        vector<int> tmp;
        for (auto i = begin(m); i != end(m); i++) {
            for (auto j = next(i); j != end(m); j++) {
                int len = abs(i->first - j->first);
                tmp.clear();
                set_intersection(begin(i->second), end(i->second),
                                 begin(j->second), end(j->second), back_inserter(tmp));
                int height = INT_MAX;
                for (int k = 1; k < tmp.size(); k++) {
                    height = min(height, tmp[k] - tmp[k-1]);
                }
                if (height < INT_MAX) res = min(res, height*len);
            }
        }
        return res == INT_MAX ? 0 : res;
    }
};
```

## 946. Validate Stack Sequences

可以很无脑的做一个栈来解决这个问题，这里巧妙利用pushed数组的前面部分模拟一个栈，以达到O(1)的空间复杂度。

```cpp
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        int i = -1, j = 0;
        for (auto k : popped) {
            while (i < 0 || pushed[i] != k) {
                if (j == pushed.size()) return false;
                pushed[++i] = pushed[j++];
            }
            i--;
        }
        return true;
    }
};
```

## 948. Bag of Tokens

贪心。

```cpp
class Solution {
public:
    int bagOfTokensScore(vector<int>& tokens, int P) {
        sort(begin(tokens), end(tokens));
        int i = 0, j = tokens.size() - 1, res = 0, r = 0;
        while (i <= j) {
            if (P >= tokens[i]) {
                P -= tokens[i++];
                r++;
                res = max(res, r);
            } else if (r > 0) {
                P += tokens[j--];
                r--;
            } else break;
        }
        return res;
    }
};
```

## 956. Tallest Billboard

定义`dp[i][d]`为用前`i`个元素，构建左右高度差为`d`时左边的最大高度，就这比较难想，然后还有一点比较怪异的是需要考虑不存在的情况，不能简单看成0，而是需要好好考虑。

```cpp
class Solution {
public:
    int tallestBillboard(vector<int>& rods) {
        if (rods.empty()) return 0;
        int sum = accumulate(begin(rods), end(rods), 0);
        vector<vector<int>> dp(rods.size(), vector<int>(sum*2+1, -1));
        dp[0][sum-rods[0]] = rods[0];
        dp[0][sum+rods[0]] = 0;
        dp[0][sum] = 0;
        for (int i = 1; i < rods.size(); i++) {
            for (int d = -sum; d <= sum; d++) {
                if (dp[i-1][sum-d] != -1) 
                    dp[i][sum-d] = dp[i-1][sum-d];
                if (sum-d-rods[i] >= 0 && dp[i-1][sum-d-rods[i]] != -1) 
                    dp[i][sum-d] = max(dp[i][sum-d], dp[i-1][sum-d-rods[i]]);
                if (sum-d+rods[i] <= 2*sum && dp[i-1][sum-d+rods[i]] != -1) 
                    dp[i][sum-d] = max(dp[i][sum-d], rods[i]+dp[i-1][sum-d+rods[i]]);
            }
        }
        return dp[rods.size()-1][sum] >= 0 ? dp[rods.size()-1][sum] : 0;
    }
};
```



## 966. Vowel Spellchecker

用三个哈希表存储，元音的替换全部转化为井号再保存。

```cpp
class Solution {
public:
    vector<string> spellchecker(vector<string>& wordlist, vector<string>& queries) {
        unordered_set<string> s;
        unordered_map<string, int> m1;
        unordered_map<string, int> m2;
        vector<string> res;
        auto toLow = [](string& a) {
            for (auto&c : a) c = tolower(c);
        };
        auto trans = [](string& a) {
            for (auto& c : a) {
                if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')
                    c = '#';
            }
        };
        for (int i = 0; i < wordlist.size(); i++) {
            auto t(wordlist[i]);
            s.insert(t);
            toLow(t);
            if (m1.find(t) == m1.end()) m1[t] = i;
            trans(t);
            if (m2.find(t) == m2.end()) m2[t] = i;
        }
        for (auto &str : queries) {
            if (s.find(str) != s.end()) res.push_back(str);
            else {
                toLow(str);
                if (m1.find(str) != m1.end()) res.push_back(wordlist[m1[str]]);
                else {
                    trans(str);
                    if (m2.find(str) != m2.end()) res.push_back(wordlist[m2[str]]);
                    else res.push_back("");
                }
            }
        }
        return res;
    }
};
```

## 976. 三角形的最大周长

排序。

```cpp
class Solution {
public:
    int largestPerimeter(vector<int>& A) {
        sort(rbegin(A), rend(A));
        for (int i = 0; i < A.size()-2; i++) {
            if (A[i]-A[i+1] < A[i+2]) return A[i]+A[i+1]+A[i+2]; 
        }
        return 0;
    }
};
```



## 998. Maximum Binary Tree II

递归地插入。如果当前最大，则将root作为左孩子，否则尝试修改root的右子树。

```cpp
class Solution {
public:
    TreeNode* insertIntoMaxTree(TreeNode* root, int val) {
        auto t = new TreeNode(val);
        if (!root) return t;
        if (val > root->val) {
            t->left = root;
            return t;
        }
        root->right = insertIntoMaxTree(root->right, val);
        return root;
    }
};
```

## 1007. Minimum Domino Rotations For Equal Row

计数每个数字在不同位置出现的次数。在每个位置，`A[i]`或`B[i]`至少要出现`i`次。

```cpp
class Solution {
public:
    int minDominoRotations(vector<int>& A, vector<int>& B) {
        unordered_map<int, int> cnt;
        for (int i = 0; i < A.size(); i++) {
            if (A[i] == B[i]) {
                if (cnt[A[i]]++ < i) return -1;
            } else {
                bool f1 = cnt[A[i]]++ < i;
                bool f2 = cnt[B[i]]++ < i;
                if (f1 && f2) return -1;
            }
        }
        int n = cnt[A[0]] == A.size() ? A[0] : B[0];
        int t1 = count(A.begin(), A.end(), n);
        int t2 = count(B.begin(), B.end(), n);
        return A.size() - max(t1, t2);
    }
};
```

## 1019. Next Greater Node In Linked List

类似496题，用栈。

```cpp
class Solution {
public:
    vector<int> nextLargerNodes(ListNode* head) {
        stack<pair<int, int>> s;
        vector<int> res;
        int i = 0;
        while (head) {
            while (!s.empty() && s.top().first < head->val) {
                if (res.size() <= s.top().second) res.resize(s.top().second+1);
                res[s.top().second] = head->val;
                s.pop();
            }
            s.push({head->val, i});
            i++;
            head = head->next;
        }
        while (!s.empty()) {
            if (res.size() <= s.top().second) res.resize(s.top().second+1);
            res[s.top().second] = 0;
            s.pop();
        }
        return res;
    }
};
```

## 1024. Video Stitching

求区间的最小覆盖。有一瞬间想到集合覆盖，以为是个NP难问题，但对于区间来说想了一下问题还是P的。考虑覆盖区间`[0,i]`的最小覆盖数，如果存在一个区间覆盖了`[A,i]`，那么我们只需归约到`[0,A]`的最小覆盖数加1即可。于是这个问题就是一个简单的DP，最终结果是所有可能的区间选择中取最小者，复杂度为$O(n^2)$。

```cpp
class Solution {
public:
    int videoStitching(vector<vector<int>>& clips, int T) {
        vector<int> dp(T+1, 10000);
        dp[0] = 0;
        for (int i = 1; i <= T; i++) {
            for (auto& c : clips) {
                if (c[0] < i && c[1] >= i)
                    dp[i] = min(dp[i], dp[c[0]]+1);
            }
        }
        return dp[T] >= 10000 ? -1 : dp[T];
    }
};
```

## 1025. Divisor Game

DP。

```cpp
class Solution {
public:
    bool divisorGame(int N) {
        vector<bool> dp(N+1, false);
        for (int i = 2; i <= N; i++) {
            for (int k = 1; k < i; k++) {
                if (i%k==0 && !dp[i-k]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[N];
    }
};
```

实际上，拿到奇数的那个人必输。证明如下：

* 首先，假如拿到N的人输，那么拿到N+1的人必胜，因为那个人只要把N+1减1给对方即可。
* 假如一个人拿到奇数，其所有因子都是奇数，因此下一轮的人必然拿到偶数。
* 归纳法证明奇数的人输，偶数的人胜：首先1输2胜；假设1-2K满足条件，对于拿到2K+1的人，无论怎么选择策略，下一轮对手必然拿到1-2K中的偶数，因此必输。据第一条结论，拿到2K+2的人就赢了。证毕。

## 1031. Maximum Sum of Two Non-Overlapping Subarrays

前缀和来保存，这样子段和可以通过减法得出，然后保存当前段之前的最大L和M子段和，遍历一次即可。

```cpp
class Solution {
public:
    int maxSumTwoNoOverlap(vector<int>& A, int L, int M) {
        vector<int> s(A);
        for (int i = 1; i < s.size(); i++) s[i] += s[i-1];
        int maxL = s[L-1], maxM = s[M-1], res = s[L+M-1];
        for (int i = L+M; i < s.size(); i++) {
            maxL = max(maxL, s[i-M]-s[i-M-L]);
            maxM = max(maxM, s[i-L]-s[i-M-L]);
            res = max(res, max(s[i]-s[i-L]+maxM, s[i]-s[i-M]+maxL));
        }
        return res;
    }
};
```

## 1032. Stream of Characters

按单词的逆序构建Trie树。

```cpp
class StreamChecker {
    struct Node {
        Node* kids[26];
        bool leaf;
        Node(bool f = false): leaf(f) { fill(kids, kids+26, nullptr); }
        ~Node() { for (auto kid : kids) delete kid; }
    };
    Node root;
    string cache;
public:
    StreamChecker(vector<string>& words): root(false) {
        for (auto const& word : words) {
            Node* r = &root;
            for (auto c = word.crbegin(); c != word.crend(); c++) {
                if (!r->kids[*c-'a']) r->kids[*c-'a'] = new Node();
                r = r->kids[*c-'a'];
            }
            r->leaf = true;
        }
    }
    
    bool query(char letter) {
        cache += letter;
        Node* p = &root;
        for (auto c = cache.rbegin(); c != cache.rend(); c++) {
            p = p->kids[*c-'a'];
            if (!p) return false;
            if (p->leaf) return true;
        }
        return false;
    }
};

/**
 * Your StreamChecker object will be instantiated and called as such:
 * StreamChecker* obj = new StreamChecker(words);
 * bool param_1 = obj->query(letter);
 */
```

## 1039.  Minimum Score Triangulation of Polygon

需要考虑一个凸多边形的所有三角形划分，从一条边起，选择另一个点，然后取出这个三角形，在递归求解剩下的两个凸多边形即可。有点像矩阵乘法的分解。

```cpp
class Solution {
public:
    int minScoreTriangulation(vector<int>& A) {
        int n = A.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int i = n-1; i >= 0; i--) {
            for (int j = i + 2; j < n; j++) {
                int t = INT_MAX;
                for (int k = i + 1; k <= j-1; k++) {
                    t = min(t, dp[i][k] + dp[k][j] + A[i]*A[k]*A[j]);
                }
                dp[i][j] = t;
            }
        }
        return dp[0][n-1];
    }
};
```

## 1078. Occurrences After Bigram[string]

简单的字符串比较。

```cpp
class Solution {
public:
    vector<string> findOcurrences(string text, string first, string second) {
        vector<string> res;
        string t;
        text += ' ';
        pair<bool, bool> p1(false, false), p2(false, false);
        for (auto c : text) {
            if (c == ' ') {
                if (p1.first && p2.second) res.push_back(t);
                p1 = p2;
                p2.first = t == first;
                p2.second = t == second;
                t.clear();
            } else t += c;
        }
        return res;
    }
};
```

## 1079. Letter Tile Possibilities

深搜，注意每一个位置只能填充不重复的子母。

```cpp
class Solution {
public:
    int numTilePossibilities(string tiles) {
        int count[26];
        fill(count, count+26, 0);
        for (auto t : tiles) count[t-'A']++;
        int res = 0;
        function<void(int)> dfs = [&](int i) {
            for (auto& k : count) {
                if (k > 0) {
                    k--;
                    dfs(i+1);
                    k++;
                }
            }
            if (i > 0) res++;
        };
        dfs(0);
        return res;
    }
};
```

## 1094. Car Pooling

按头排序，按尾入最小堆。

```cpp
class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        sort(begin(trips), end(trips), [](vector<int>& a, vector<int>& b) {
            return a[1] < b[1];
        });
        auto cmp = [](vector<int>& a, vector<int>& b) {
            return a[2] > b[2];
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> pq(cmp);
        int sz = 0;
        for (auto &trip : trips) {
            while (!pq.empty() && pq.top()[2] <= trip[1]) {
                sz -= pq.top()[0];
                pq.pop();
            }
            pq.push(trip);
            sz += trip[0];
            if (sz > capacity) return false;
        }
        return true;
    }
};
```

由于题目限制，可以直接记录每个位置的下人和上人数量，然后遍历所有1000个位置即可，但是这吃测试数据吧。还有一种方法，直接将所有头尾排序，然后遍历一遍，遇头就加，遇尾就减即可，这个更合理，复杂度都是$O(n\lg n)$。

## 1096. Brace Expansion II

递归进行parse，按照逗号的出现区分是进行笛卡尔积还是取并集。总之需要十分仔细。

```cpp
class Solution {
    vector<string> parse(string& s, int& i) {
        vector<string> res;
        vector<string> tmp;
        while (true) {
            tmp.clear();
            while (i < s.size() && s[i] != ',' && s[i] != '}') {
                if (s[i] == '{') {
                    i++;
                    auto t = parse(s, i);
                    if (tmp.empty()) tmp = std::move(t);
                    else {
                        vector<string> tmp_;
                        for (auto &s1 : tmp)
                            for (auto &s2 : t)
                                tmp_.push_back(s1+s2);
                        tmp = std::move(tmp_);
                    }
                } else {
                    string t;
                    while (s[i] >= 'a' && s[i] <= 'z') t += s[i++];
                    if (tmp.empty()) tmp.push_back(t);
                    else {
                        vector<string> tmp_;
                        for (auto &s : tmp) tmp_.push_back(s+t);
                        tmp = std::move(tmp_);
                    }
                }
            }
            copy(tmp.begin(), tmp.end(), back_inserter(res));
            if (i == s.size() || s[i] == '}') break;
            i++;
        }
        i++;
        sort(res.begin(), res.end());
        res.erase(unique(res.begin(), res.end()), res.end());
        return res;
    }
public:
    vector<string> braceExpansionII(string expression) {
        int i = 0;
        return parse(expression, i);
    }
};
```

## 1109. Corporate Flight Bookings

给定一个三元组的数组`[i,j,k]`，将给结果数组的i至j累加k，如何线性求解结果数组？在起始位置i设置累加值为k，在结束位置j+1设置累加值为-k，这样即可实现只在[i,j]之间的位置累加k。巧妙。

```cpp
class Solution {
public:
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> res(n, 0);
        for (auto& b : bookings) {
            res[b[0] - 1] += b[2];
            if (b[1] < n)
                res[b[1]] -= b[2];
        }
        for (int i = 1; i < n; i++) res[i] += res[i-1];
        return res;
    }
};
```

## 1124. Longest Well-Performing Interval

要求和为正的最长连续子列，当然子列中只存在+1和-1。依然是求前缀和，如果前缀和大于0，显然直接更新结果即可，否则，我们需要考虑是否存在之前的一个前缀和小于当前的，这样相减即可。例如当前和为-1，那就要找最先出现的前缀和小于-1的位置。由于题目条件，实际上我们只需要找前缀和等于-2的位置即可，为什么呢？假如存在等于-3的位置，那由于一开始是0，变化过程中肯定有一个-2在-3的前面，因此我们只需寻找小1的最早出现位置。这里最早出现可用哈希表来保存，只保存第一次出现的位置。

```cpp
class Solution {
public:
    int longestWPI(vector<int>& hours) {
        unordered_map<int, int> m;
        int sum = 0, res = 0;
        for (int i = 0; i < hours.size(); i++) {
            sum += hours[i] > 8 ? 1 : -1;
            if (sum > 0) res = max(res, i+1);
            else {
                if (m.find(sum-1) != m.end()) res = max(res, i-m[sum-1]);
                if (m.find(sum) == m.end()) m[sum] = i;
            }
        }
        return res;
    }
};
```

## 1139. Largest 1-Bordered Square

先通过DP计算每个点往上和往右最多延申多远，然后判断正方形。

```cpp
class Solution {
public:
    int largest1BorderedSquare(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        vector<vector<pair<int,int>>> dp(n, vector<pair<int, int>>(m, {0,0}));
        int res = -1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) {
                    res = 0;
                    dp[i][j].first = i > 0 ? dp[i-1][j].first + 1 : 1;
                    dp[i][j].second = j > 0 ? dp[i][j-1].second + 1 : 1;
                }
            }
        }
        if (res < 0) return 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) {
                    for (int k = res; i+k<n&&j+k<m; k++) {
                        if (dp[i+k][j+k].first >= k+1 && dp[i+k][j+k].second >= k+1 &&
                            dp[i][j+k].second >= k+1 && dp[i+k][j].first >= k+1)
                            res = k;
                    }
                }
            }
        }
        return (res+1)*(res+1) ;
    }
};
```

## 1157. Online Majority Element In Subarray

获取区间内的多数元素（threshold的大小限制），可以用简单的随机算法（因为多数元素必然大于一半，所以从区间内任选1个元素，它是多数元素的概率大于二分之一）。下面只需要判断这个元素是否是多数的，我们用二分查找判断当前区间内此元素出现次数即可。

保存从数到index的哈希表，保持index是升序的，那么就可以方便地二分查left和right了。

```go
type MajorityChecker struct {
	nums    []int
	num2idx map[int][]int
}

func Constructor(arr []int) MajorityChecker {
	var res MajorityChecker
	res.nums = arr
	res.num2idx = make(map[int][]int)
	for i, n := range arr {
		res.num2idx[n] = append(res.num2idx[n], i)
	}
	return res
}

func (this *MajorityChecker) Query(left int, right int, threshold int) int {
	var t, l, r int
	for _i := 0; _i < 10; _i++ {
		t = this.nums[rand.Intn(right - left + 1)+left]
		l = sort.Search(len(this.num2idx[t]), func(i int) bool {
			return this.num2idx[t][i] >= left
		})
		r = sort.Search(len(this.num2idx[t]), func(i int) bool {
			return this.num2idx[t][i] > right
		})
		if r-l >= threshold {
			return t
		}
	}
	return -1
}
```

Rust标准库没有随机数，故遍历之，但是依然快了2倍，rust牛逼！

```rust
use std::collections::HashMap;

struct MajorityChecker {
    map: HashMap<i32, Vec<usize>>,
}

fn b_search(arr: &Vec<usize>, n: usize, flag: bool) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len() - 1;
    let mut mid: usize = 0;
    while lo < hi {
        mid = lo + (hi - lo) / 2;
        if flag { // >=
            if arr[mid] >= n {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        } else { // >
            if arr[mid] > n {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
    }
    if flag {
        lo
    } else {
        if arr[lo] > n {
            lo
        } else {
            lo + 1
        }
    }
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MajorityChecker {

    fn new(arr: Vec<i32>) -> Self {
        let mut map: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, n) in arr.iter().enumerate() {
            match map.get_mut(n) {
                Some(num) => num.push(i),
                None => {
                    map.insert(*n, vec![i]);
                }
            }
        }
        MajorityChecker{map}
    }

    fn query(&self, left: i32, right: i32, threshold: i32) -> i32 {
        for (n, idx) in self.map.iter() {
            if idx.len() < threshold as usize {
                continue;
            }
            let l = b_search(idx, left as usize, true);
            let r = b_search(idx, right as usize, false);
            if r - l >= threshold as usize {
                return *n;
            }
        }
        -1
    }
}
```



## 1177. Can Make Palindrome from Substring

重点是如果出现的奇数次的字母有2个，那么我们可以将其中一个字母变成另一个，这样两个都变成偶数次。

```cpp
class Solution {
public:
    vector<bool> canMakePaliQueries(string s, vector<vector<int>>& queries) {
        vector<bool> res;
        vector<vector<int>> count(s.size(), vector<int>(26, 0));
        for (int i = 0; i < s.size(); i++) {
            if (i == 0) count[i][s[i]-'a']++;
            else {
                for (int j = 0; j < 26; j++) count[i][j] = count[i-1][j];
                count[i][s[i]-'a']++;
            }
        }
        for (auto &q: queries) {
            int odds = 0;
            int l = q[0], r = q[1], k = q[2];
            for (int i = 0; i < 26; i++) {
                if (l == 0) odds += count[r][i]&1;
                else odds += (count[r][i]-count[l-1][i])&1;
            }
            if (odds/2 <= k) res.push_back(true);
            else res.push_back(false);
        }
        return res;
    }
};
```

## 1178. Number of Valid Words for Each Puzzle

本题关键在于如何在第二次查找时不遍历整个word数组，首先是利用哈希表，把word的mask存起来；但还不够，再利用puzzle的长度为7这一特点，而且第一个字母必须出现在对应的word上，因此符合条件的word数量就限制在了$2^6=64$这个量级，用深搜去找这些可能的mask组合，这样就把复杂度降到线性了。

```cpp
class Solution {
    int dfs(string& puzzle, int mask, int i, map<int, int> &m) {
        if (i == 7) {
            auto i = m.find(mask);
            return i == m.end() ? 0 : i->second;
        }
        return dfs(puzzle, mask, i+1, m) + dfs(puzzle, mask | (1<<(puzzle[i]-'a')), i+1, m);
    }
public:
    vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles) {
        vector<int> res(puzzles.size(), 0);
        map<int, int> m;
        for (int i = 0; i < words.size(); i++) {
            int t = 0;
            for (auto c : words[i]) t |= 1 << (c-'a');
            m[t]++;
        }
        for (int i = 0; i < puzzles.size(); i++) {
            res[i] += dfs(puzzles[i], 1<<(puzzles[i][0]-'a'), 1, m);
        }
        return res;
    }
};
```

## 1203. Sort Items by Groups Respecting Dependencies

挺难一题，不过我居然一次过了哈哈。需要进行两次拓扑排序，第一次是group间的，第二次是对每一个group之中的item进行排序。因此首先的处理是把同一组的item组合在一起，方便后续，同时我们考虑将无group的item单独形成一个group。

而后，按照group间的依赖关系生成拓扑序，之后按照这个拓扑序分别对每组内部的元素再进行拓扑排序放在一起，这样生成的顺序既不违背item间的拓扑序，也不违背同组的item必须排在一起的限制。在任何一次拓扑排序中遇到环都说明不存在有效的排列结果，直接输出空集就可。

```cpp
class Solution {
public:
    vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
        vector<vector<int>> groups(m, vector<int>());
        // group items together
        for (int i = 0; i < n; i++) {
            if (group[i] == -1) {
                group[i] = groups.size();
                groups.push_back({i});
            } else {
                groups[group[i]].push_back(i);
            }
        }
        m = groups.size();
        vector<int> group_sorted;
        vector<int> v1(m, 0);
        // group-wise topological sort
        function<bool(int)> dfs = [&](int i) {
            v1[i] = 1;
            for (auto k : groups[i]) {
                for (auto j : beforeItems[k]) {
                    if (group[j] != i) {
                        if (v1[group[j]] == 1) return false;
                        if (v1[group[j]] == 0 && !dfs(group[j])) return false; 
                    }
                }
            }
            group_sorted.push_back(i);
            v1[i] = 2;
            return true;
        };
        for (int i = 0; i < m; i++) {
            if (v1[i] == 0 && !dfs(i))
                return {};
        }
        vector<int> res;
        vector<int> v2(n, 0);
        // item-wise topological sort, group by group
        function<bool(int)> dfs_in_group = [&](int i) {
            v2[i] = 1;
            for (auto j : beforeItems[i]) {
                if (group[j] == group[i]) {
                    if (v2[j] == 1) return false;
                    if (v2[j] == 0 && !dfs_in_group(j)) return false;
                }
            }
            res.push_back(i);
            v2[i] = 2;
            return true;
        };
        for (auto i : group_sorted) {
            if (groups[i].size() == 1) res.push_back(groups[i][0]);
            else {
                for (auto k : groups[i]) {
                    if (v2[k] == 0 && !dfs_in_group(k))
                        return {};
                }
            }
        }
        return res;
    }
};
```

## 1233. Remove Sub-Folders from the Filesystem

排序最简单，不过我做了个trie也能过吧。

```cpp
class Solution {
    struct node {
        vector<node*> kids;
        bool isFolder;
        node(): isFolder(false), kids(27, nullptr) {}
        ~node() { for (auto t : kids) delete t; }
        void calc(vector<string>& r, string t, bool found) {
            if (isFolder) {
                r.push_back(t);
                found = true;
            }
            for (int i = 0; i < 26; i++) {
                if (kids[i]) kids[i]->calc(r, t+char('a'+i), found);
            }
            if (kids[26] && !found) kids[26]->calc(r, t+'/', found);
        }
    };
public:
    vector<string> removeSubfolders(vector<string>& folder) {
        node* root = new node();
        for (auto &s : folder) {
            node* cur = root;
            for (auto c : s) {
                int id = c == '/' ? 26 : c - 'a';
                if (!cur->kids[id]) cur->kids[id] = new node();
                cur = cur->kids[id];
            }
            cur->isFolder = true;
        }
        vector<string> res;
        root->calc(res, "", false);
        delete root;
        return res;
    }
};
```

## 1255. Maximum Score Words Formed by Letters

很普通的深搜。

```cpp
class Solution {
public:
    int maxScoreWords(vector<string>& words, vector<char>& letters, vector<int>& score) {
        int c[26] = {0};
        for (auto x : letters) c[x-'a']++;
        vector<vector<int>> w(words.size(), vector<int>(26, 0));
        for (int i = 0; i < words.size(); i++) {
            for (auto c : words[i]) w[i][c-'a']++;
        }
        int res = 0;
        int t[26] = {0};
        function<void(int)> dfs = [&](int i) {
            for (int k = 0; k < 26; k++) {
                if (t[k] > c[k]) return;
            }
            if (i == words.size()) {
                int s = 0;
                for (int k = 0; k < 26; k++) {
                    s += t[k]*score[k];
                }
                if (s > res) res = s;
                return;
            }
            dfs(i+1);
            for (int k = 0; k < 26; k++) t[k] += w[i][k];
            dfs(i+1);
            for (int k = 0; k < 26; k++) t[k] -= w[i][k];
        };
        dfs(0);
        return res;
    }
};
```

## 1263. Minimum Moves to Move a Box to Their Target Location

推箱子游戏，并没什么好办法。通过BFS维护当前箱子和玩家的状态，尝试向四个方向推箱子，这一步通过BFS来确定当前玩家位置能否到达推箱子的位置，否则不能推动。此外，第一层BFS需要避免重复状态搜索，即向同一个方向推相同位置的箱子，这一重复状态通过一个哈希表来记录。

```cpp
class Solution {
    struct Pos {
        int bx;
        int by;
        int sx;
        int sy;
        int cost;
        int hash;
        Pos(int a, int b, int c, int d, int e)
            :bx(a),by(b),sx(c),sy(d),cost(e){}
    };
public:
    int minPushBox(vector<vector<char>>& grid) {
        int sx, sy, bx, by, tx, ty, m = grid.size(), n = grid[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 'S') {
                    sx = i; sy = j;
                } else if (grid[i][j] == 'T') {
                    tx = i; ty = j;
                } else if (grid[i][j] == 'B') {
                    bx = i; by = j;
                }
            }
        }
        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};
        function<bool(Pos,int,int)> canReach = [&](Pos p, int x, int y) {
            if (grid[x][y] == '#') return false;
            auto tmp = grid[p.bx][p.by];
            grid[p.bx][p.by] = '#';
            vector<vector<int>> v(m, vector<int>(n, 0));
            queue<pair<int, int>> q({make_pair(p.sx, p.sy)});
            v[p.sx][p.sy] = 1;
            while (!q.empty()) {
                auto t = q.front();
                q.pop();
                for (int i = 0; i < 4; i++) {
                    int nx = t.first + dx[i];
                    int ny = t.second + dy[i];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && !v[nx][ny] && grid[nx][ny] != '#') {
                        if (nx == x && ny == y) {
                            grid[p.bx][p.by] = tmp;
                            return true;
                        }
                        v[nx][ny] = 1;
                        q.push({nx, ny});
                    }
                }
            }
            grid[p.bx][p.by] = tmp;
            return false;
        };
        queue<Pos> q({Pos(bx, by, sx, sy, 0)});
        unordered_set<int> s;
        while (!q.empty()) {
            auto p = q.front();
            q.pop();
            if (p.bx == tx && p.by == ty) return p.cost;
            for (int i = 0; i < 4; i++) {
                int nx = p.bx + dx[i];
                int ny = p.by + dy[i];
                int rx = p.bx - dx[i];
                int ry = p.by - dy[i];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] != '#') {
                    if (rx >= 0 && rx < m && ry >= 0 && ry < n && canReach(p, rx, ry)) {
                        auto next = Pos(nx, ny, rx, ry, p.cost+1);
                        int k = i|((nx*n+ny)<<3);
                        if (s.find(k) == s.end()) {
                            s.insert(k);
                            q.push(next);
                        }
                    }
                }
            }
        }
        return -1;
    }
};
```

## 1277. Count Square Submatrices with All Ones

DP计数以格子`[i][j]`为右下角元素的方阵的最大边长，则`dp[i][j]=min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1`。

```cpp
class Solution {
public:
    int countSquares(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] != 0) {
                    if (i == 0 || j == 0) dp[i][j] = 1;
                    else dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1;
                    res += dp[i][j];
                }
            }
        }
        return res;
    }
};
```



## 1291. Sequential Digits

转化成字符串，然后深搜填充夹在中间的数。感觉这是个范式。。。类似的题目都可以这么做。

```cpp
class Solution {
public:
    vector<int> sequentialDigits(int low, int high) {
        vector<int> res;
        string lo = to_string(low);
        string hi = to_string(high);
        while (lo.size() < hi.size()) lo.insert(0, "0");
        string tmp;
        function<void(int,bool,bool)> dfs = [&](int i, bool same1, bool same2) {
            if (i == lo.size()) {
                res.push_back(stoi(tmp));
                return;
            }
            char s = same1 ? lo[i] : '0';
            char e = same2 ? hi[i] : '9';
            if (tmp.empty() || tmp.back() == '0') {
                for (char c = s; c <= e; c++) {
                    tmp.push_back(c);
                    dfs(i+1, same1&&c==lo[i], same2&&c==hi[i]);
                    tmp.pop_back();
                }
            } else if (tmp.back()+1 >= s && tmp.back()+1 <= e) {
                char c = tmp.back()+1;
                tmp.push_back(c);
                dfs(i+1, same1&&c==lo[i], same2&&c==hi[i]);
                tmp.pop_back();
            }
        };
        dfs(0, true, true);
        sort(res.begin(), res.end());
        return res;
    }
};
```

## 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

方阵和可以通过左上角矩形的和加减得出，因此先计算sum矩阵，直接覆盖mat即可。然后我们对边长二分查找，确定最大边长。复杂度为$O(n^2)$。

```cpp
class Solution {
public:
    int maxSideLength(vector<vector<int>>& mat, int threshold) {
        int m = mat.size(), n = mat[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j > 0) mat[i][j] += mat[i][j-1];
                if (i > 0) mat[i][j] += mat[i-1][j];
                if (i > 0 && j > 0) mat[i][j] -= mat[i-1][j-1];
            }
        }
        int lo = 0, hi = min(m, n) + 1;
        while (hi - lo > 1) {
            int mid = lo + (hi-lo)/2;
            bool flag = false;
            for (int i = mid-1; i < m; i++) {
                for (int j = mid-1; j < n; j++) {
                    int sum = mat[i][j];
                    if (i - mid >= 0) sum -= mat[i-mid][j];
                    if (j - mid >= 0) sum -= mat[i][j-mid];
                    if (i >= mid && j >= mid) sum += mat[i-mid][j-mid];
                    if (sum <= threshold) {
                        flag = true;
                        break;
                    }
                }
                if (flag) break;
            }
            if (flag) lo = mid;
            else hi = mid;
        }
        return lo;
    }
};
```

##  1296. Divide Array in Sets of K Consecutive Numbers

计数然后暴力：

```cpp
class Solution {
public:
    bool isPossibleDivide(vector<int>& nums, int k) {
        sort(begin(nums), end(nums));
        deque<int> t;
        for (int i = 0; i < nums.size(); i++) {
            if (t.empty()) t.push_back(1);
            else if (nums[i] == nums[i-1]) t.back()++;
            else if (t.size() == k) {
                int d = t.front();
                for (auto& x : t) x -= d;
                while (!t.empty() && t.front() <= 0) {
                    if (t.front() < 0) return false;
                    else t.pop_front();
                }
                i--;
            } else {
                if (nums[i] == nums[i-1] + 1) t.push_back(1);
                else return false;
            }
        }
        if (!t.empty()) {
            if (t.size() < k) return false;
            for (auto x : t) if (x != t.front()) return false;
        }
        return true;
    }
};
```

聪明的方法：按k的余数分别计数，所有数量都相等，则可以分割。因为连续的k个数除以k的余数一定是0,1..k-1。

## 1297. Maximum Number of Occurrences of a Substring

关键点在于只要计算最短长度的出现次数最大值即可，因为较长的多次出现必然意味着其子串的多次出现。所以这里的maxSize是混淆视听的参数。

```cpp
class Solution {
public:
    int maxFreq(string s, int maxLetters, int minSize, int maxSize) {
        unordered_map<string, int> m;
        int res = 0;
        vector<int> c(26, 0);
        for (int i = 0; i < s.size() - minSize + 1; i++) {
            string t(s.substr(i, minSize));
            int count = 0;
            fill(begin(c), end(c), 0);
            for (auto ch : t) {
                if (++c[ch-'a'] == 1) count++;
            }
            if (count <= maxLetters) {
                if ((count = ++m[t]) > res) res = count;
            }
        }
        return res;
    }
};
```



## 1300. Sum of Mutated Array Closest to Target

先排序，而后从小到大尝试。也可以二分查找，但查找过程中还得按照值来遍历数组求和，复杂度没降。

```cpp
class Solution {
public:
    int findBestValue(vector<int>& arr, int target) {
        sort(arr.begin(), arr.end());
        int sum = 0, k = 0, n = arr.size(), diff = INT_MAX;
        for (int i = 0; i <= arr.back(); i++) {
            if (i >= arr[k]) {
                sum += arr[k];
                k++;
            }
            int ndiff = abs(target-i*(n-k)-sum);
            if (diff > ndiff) {
                diff = ndiff;
            } else return i-1;
            if (k == n) return arr[k-1];
        }
        return -1;
    }
};
```



## 1340. Jump Game V

转成DAG，然后深搜找最长路。其实不需要做成图的，不过一样就是了。

```cpp
class Solution {
public:
    int maxJumps(vector<int>& arr, int d) {
        int n = arr.size();
        vector<vector<int>> adj(n, vector<int>());
        for (int i = 0; i < n; i++) {
            for (int x = 1; x <= d; x++) {
                if (i+x < n && arr[i+x] < arr[i]) {
                    adj[i].push_back(i+x);
                } else break;
            }
            for (int x = 1; x <= d; x++) {
                if (i-x >= 0 && arr[i-x] < arr[i]) {
                    adj[i].push_back(i-x);
                } else break;
            }
        }
        vector<int> mem(n, -1);
        int res = 0;
        for (int i = 0; i < n; i++) {
            if (mem[i] == -1) dfs(mem, adj, i, res);
        }
        return res;
    }
    int dfs(vector<int>& mem, vector<vector<int>>& adj, int i, int& res) {
        if (mem[i] != -1) return mem[i];
        int nextmax = 0;
        for (auto j: adj[i]) {
            nextmax = max(nextmax, dfs(mem, adj, j, res));
        }
        mem[i] = nextmax + 1;
        res = max(res, mem[i]);
        return mem[i];
    }
};
```

## 1345. Jump Game IV

BFS找最短路径。

```cpp
class Solution {
public:
    int minJumps(vector<int>& arr) {
        int n = arr.size();
        if (n == 1) return 0;
        unordered_map<int, vector<int>> num2id;
        for (int i = n - 1; i >= 0; i--) {
            num2id[arr[i]].push_back(i);
        }
        vector<bool> visited(n, false);
        vector<int> dist(n, INT_MAX);
        queue<int> ids;
        ids.push(0);
        dist[0] = 0;
        visited[0] = true;
        while (!ids.empty()) {
            int i = ids.front();
            ids.pop();
            if (i-1 >= 0 && !visited[i-1]) {
                ids.push(i-1);
                dist[i-1] = dist[i] + 1;
                visited[i-1] = true;
            }
            if (i+1 < n && !visited[i+1]) {
                ids.push(i+1);
                dist[i+1] = dist[i] + 1;
                visited[i+1] = true;
                if (i+1 == n-1) return dist[n-1];
            }
            for (auto j : num2id[arr[i]]) {
                if (j != i && j != i-1 && j != i+1 && !visited[j]) {
                    ids.push(j);
                    dist[j] = dist[i] + 1;
                    visited[j] = true;
                    if (j == n-1) return dist[n-1];
                }
            }
        }
        return -1;
    }
};
```

## 1361. Validate Binary Tree Nodes

确定根结点然后深搜，正确的二叉树不会重复访问，且必然连通。

```cpp
class Solution {
public:
    bool validateBinaryTreeNodes(int n, vector<int>& leftChild, vector<int>& rightChild) {
        vector<bool> visited(n, false);
        for (int i = 0; i < n; i++) {
            if (leftChild[i] != -1) visited[leftChild[i]] = true;
            if (rightChild[i] != -1) visited[rightChild[i]] = true;
        }
        int root = -1;
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                root = i;
                break;
            }
        }
        if (root < 0) return false;
        fill(visited.begin(), visited.end(), false);
        bool res = true;
        function<void(int)> dfs = [&](int i) {
            if (visited[i]) {
                res = false;
                return;
            }
            visited[i] = true;
            if (leftChild[i] != -1) dfs(leftChild[i]);
            if (rightChild[i] != -1) dfs(rightChild[i]);
        };
        dfs(root);
        for (auto f : visited) {
            if (!f) res = false;
        }
        return res;
    }
};
```

## 1362. Closest Divisors

从平方根加一开始测试是否整除。不要被双变量迷惑把问题想复杂。

```cpp
class Solution {
public:
    vector<int> closestDivisors(int num) {
        int root = sqrt(num) + 1;
        while (true) {
            if ((num + 1) % root == 0) {
                return {root, (num+1)/root};
            } else if ((num + 2) % root == 0) {
                return {root, (num+2)/root};
            } else {
                root--;
            }
        }
        return {-1,-1};
    }
};
```

## 1363. Largest Multiple of Three

3的倍数等价于所有位加起来是3的倍数。先降序排列成可能的最大整数，而后去掉最小的digit使得模3为零。

```cpp
class Solution {
public:
    string largestMultipleOfThree(vector<int>& digits) {
        sort(digits.begin(), digits.end(), [](int a, int b) {
            return a > b;
        });
        vector<vector<int>> mem(3, vector<int>());
        int mod = 0;
        for (int i = 0; i < digits.size(); i++) {
            mod = (mod + digits[i]) % 3;
            mem[digits[i] % 3].push_back(i);
        }
        unordered_set<int> toerase;
        if (mod != 0) {
            if (!mem[mod].empty()) toerase.insert(*mem[mod].rbegin());
            else {
                toerase.insert(*mem[3-mod].rbegin());
                toerase.insert(*(mem[3-mod].rbegin() + 1));
            }
        }
        string res;
        for (int i = 0; i < digits.size(); i++) {
            if (toerase.find(i) == toerase.end()) res.push_back('0'+digits[i]);
        }
        if (res.size() == 0) return "";
        int start = 0;
        while (start != res.size() && res[start] == '0') start++;
        if (start == res.size()) return "0";
        return res.substr(start);
    }
};
```

## 1367. Linked List in Binary Tree

遍历二叉树，保存当前路径，判断目标路径是否是当前路径的后缀。

```cpp
class Solution {
public:
    bool isSubPath(ListNode* head, TreeNode* root) {
        vector<int> path;
        vector<int> target;
        while (head) {
            target.push_back(head->val);
            head = head->next;
        }
        function<bool(TreeNode*)> traverse = [&](TreeNode* t) {
            if (!t) return false;
            path.push_back(t->val);
            if (t->val == target.back() && path.size() >= target.size()) {
                auto it = path.rbegin();
                bool flag = true;
                for (int i = target.size() - 1; i >= 0; i--, it++) {
                    if (target[i] != *it) {
                        flag = false;
                        break;
                    }
                }
                if (flag) return true;
            }
            if (traverse(t->left) || traverse(t->right)) return true;
            path.pop_back();
            return false;
        };
        return traverse(root);
    }
};
```

## 1368. Minimum Cost to Make at Least One Valid Path in a Grid

如果顺着箭头则边权重为0，否则为1，转化为最短路径问题用Dijkstra解决。

```cpp
class Solution {
public:
    int minCost(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<pair<int, int>>> adj(m*n, vector<pair<int,int>>());
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i - 1 >= 0) adj[i*n+j].push_back({(i-1)*n+j, grid[i][j] != 4});
                if (i + 1 < m) adj[i*n+j].push_back({(i+1)*n+j, grid[i][j] != 3});
                if (j - 1 >= 0) adj[i*n+j].push_back({i*n+j-1, grid[i][j] != 2});
                if (j + 1 < n) adj[i*n+j].push_back({i*n+j+1, grid[i][j] != 1});
            }
        }
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> q;
        q.push({0,0});
        vector<int> d(m*n, INT_MAX);
        vector<int> visited(m*n, false);
        d[0] = 0;
        while (!q.empty()) {
            int di = q.top().first;
            int i = q.top().second;
            q.pop();
            visited[i] = true;
            if (i == m*n - 1) return di;
            for (auto p : adj[i]) {
                int j = p.first;
                if (visited[j]) continue;
                int dij = p.second;
                if (dij + di < d[j]) {
                    d[j] = di + dij;
                    q.push({d[j], j});
                }
            }
        }
        return -1;
    }
};
```

由于所有边的权重为0/1，实际上可以不用优先队列，因为此时原优先队列中距离之差不超过1，新加入的结点要么和取出的最小结点距离相同，要么大1，因此只要直接加到队列头部或尾部就可以维护优先队列的性质了。用一个双向队列替换之：

```cpp
class Solution {
public:
    int minCost(vector<vector<int>>& grid) {
        // 前面相同
        deque<int> q;
        q.push_back(0);
        vector<int> d(m*n, INT_MAX);
        vector<int> visited(m*n, false);
        d[0] = 0;
        while (!q.empty()) {
            int i = q.front();
            q.pop_front();
            visited[i] = true;
            if (i == m*n - 1) return d[i];
            for (auto p : adj[i]) {
                int j = p.first;
                if (visited[j]) continue;
                int dij = p.second;
                if (dij + d[i] < d[j]) {
                    d[j] = d[i] + dij;
                    if (dij == 1) q.push_back(j);
                    else q.push_front(j);
                }
            }
        }
        return -1;
    }
};
```

## 1373. Maximum Sum BST in Binary Tree

遍历，同时求和并判断是否符合BST的条件。

```cpp
class Solution {
public:
    int maxSumBST(TreeNode* root) {
        int res = 0;
        function<pair<int, bool>(TreeNode*)> dfs = [&](TreeNode* t) {
            if (!t) return make_pair(0, true);
            auto l = dfs(t->left);
            auto r = dfs(t->right);
            if (l.second && r.second) {
                if ((!t->left || t->val > t->left->val) 
                    && (!t->right || t->val < t->right->val)) {
                    res = max(res, l.first + r.first + t->val);
                    return make_pair(l.first + r.first + t->val, true);
                }
            }
            return make_pair(-1, false);
        };
        dfs(root);
        return res;
    }
};
```

## 1375. Bulb Switcher III

只需要判断当前位置i为止是否存在1~i的排列，用最大值判断。

```cpp
class Solution {
public:
    int numTimesAllBlue(vector<int>& light) {
        int res = 0;
        int tmpmax = -1;
        for (int i = 0; i < light.size(); i++) {
            tmpmax = max(tmpmax, light[i]);
            if (tmpmax <= i + 1) res++;
        }
        return res;
    }
};
```

## 1376. Time Needed to Inform All Employees

题目扯了半天其实就是求树的最大权重路径，深搜搞定。

```cpp
class Solution {
public:
    int numOfMinutes(int n, int headID, vector<int>& manager, vector<int>& informTime) {
        vector<vector<int>> child(n, vector<int>());
        for (int i = 0; i < n; i++) {
            if (manager[i] >= 0) child[manager[i]].push_back(i);
        }
        int res = 0;
        function<void(int, int)> dfs = [&](int i, int time) {
            if (informTime[i] == 0) res = max(time, res);
            else {
                for (auto child : child[i]) {
                    dfs(child, informTime[i] + time);
                }
            }
        };
        dfs(headID, 0);
        return res;
    }
};
```

## 1377. Frog Position After T Seconds

题目有点难理解，但其实就是在树上遍历一下就行，因为树中两结点路径是唯一的，要注意青蛙跳到非叶子结点后是不可能停留的，因此一旦路径长度小于t，则目标位置必须是叶子结点才能满足，否则就跳走了。

```cpp
class Solution {
public:
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        vector<vector<int>> adj(n + 1, vector<int>());
        for (auto e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        adj[1].push_back(0);
        vector<bool> visited(n + 1, false);
        visited[0] = true;
        double res = 0;
        function<void(int, double, int)> dfs = [&](int i, double prob, int len) {
            int n = adj[i].size() - 1;
            if (i == target && len <= t) {
                if (len == t || n == 0) res = prob;
                return;
            }
            visited[i] = true;
            for (auto j : adj[i]) {
                if (!visited[j])
                    dfs(j, prob/static_cast<double>(n), len + 1);
            }
            visited[i] = false;
        };
        dfs(1, 1.0, 0);
        return res;
    }
};
```

## 1381. Design a Stack With Increment Operation

虽然可以naive的$O(n)$实现，但是不得不说还有一个厉害的$O(1)$实现，用一个数组来表示0至i的增长量，因为栈一定是从后开始出栈，因此每次出栈时才将值加上，并且将0至i的增长量恢复到i-1，以便下一次出栈。

```cpp
class CustomStack {
    vector<int> stack, inc;
    int n;
public:
    CustomStack(int maxSize) {
        n = maxSize;
        inc.resize(n);
    }

    void push(int x) {
        if (stack.size() < n)
            stack.push_back(x);
    }

    int pop() {
        int i = stack.size() - 1;
        if (i < 0)
            return -1;
        if (i) inc[i - 1] += inc[i];
        int res = stack.back() + inc[i];
        inc[i] = 0;
        stack.pop_back();
        return res;
    }

    void increment(int k, int val) {
        int i = min(k, (int)stack.size()) - 1;
        if (i >= 0)
            inc[i] += val;
    }
};

/**
 * Your CustomStack object will be instantiated and called as such:
 * CustomStack* obj = new CustomStack(maxSize);
 * obj->push(x);
 * int param_2 = obj->pop();
 * obj->increment(k,val);
 */
```

##  1382. Balance a Binary Search Tree

从BST中构建一个有序数组，再把有序数组递归构建成平衡树即可。

```cpp
class Solution {
    void traverse(TreeNode* t, vector<int>& nums) {
        if (!t) return;
        traverse(t->left, nums);
        nums.push_back(t->val);
        traverse(t->right, nums);
    }
    TreeNode* construct(int left, int right, vector<int>& nums) {
        if (left >= right) return nullptr;
        int mid = left + (right - left)/2;
        return new TreeNode(nums[mid], construct(left, mid, nums), construct(mid+1, right, nums));
    }
public:
    TreeNode* balanceBST(TreeNode* root) {
        vector<int> nums;
        traverse(root, nums);
        return construct(0, nums.size(), nums);
    }
};
```

## 1383. Maximum Performance of a Team

将efficiency按照降序排列，这样每次新加入的worker必然效率最低，从而我们只需要计算speed之和，如果worker数量大于k，则将之前加入的speed最小的worker删去。这样每次迭代至worker[i]计算的是以第i个worker作为效率最小者的至多k个worker的最好performance，最优解就在这n个值中产生。

```cpp
class Solution {
public:
    int maxPerformance(int n, vector<int>& speed, vector<int>& efficiency, int k) {
        vector<pair<int, int>> eff_sp;
        priority_queue <int, vector<int>, greater<int>> pq_speeds;
        long sum = 0, res = 0;
        for (auto i = 0; i < n; ++i)
            eff_sp.push_back({efficiency[i], speed[i]});
        sort(begin(eff_sp), end(eff_sp));
        for (auto i = n - 1; i >= 0; --i) {
            sum += eff_sp[i].second;
            pq_speeds.push(eff_sp[i].second);
            if (pq_speeds.size() > k) {
                sum -= pq_speeds.top();
                pq_speeds.pop();
            }
            res = max(res, sum * eff_sp[i].first);
        }
        return res % 1000000007;
    }
};
```

## 1388. Pizza With 3n Slices

很简单一题，被我想到沟里去了。。。由于是循环数组，如果我们不考虑头尾相接的话，就有可能产生同时取头尾的非法解。但是考虑循环的话问题就复杂了。因此简单的方法：强迫把头或尾设置为零，这样就不可能同时选择头尾。对新产生的两个数组运行一遍非循环数组中取最大不相邻集的算法，比较即可。

至于在非循环数组中取最大独立集，很简单的DP即可解决，设$dp[i][k]$为数组$[0:i]$中取$k$个不相邻元素的最大值。则要么取$i$，要么不取$i$：$dp[i][k]=max(dp[i-1][k],dp[i-2][k-1]+a[i])$。

```cpp
class Solution {
    int maxWithoutCircle(vector<int>& s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n/3+1, 0));
        for (int k = 1; k <= n/3; k++) {
            for (int i = 2*k - 2; i < n; i++) {
                if (i == 0) dp[0][1] = s[0];
                else if (i == 1) dp[1][1] = max(s[0], s[1]);
                else {
                    dp[i][k] = max(dp[i-1][k], dp[i-2][k-1] + s[i]);  
                }
            }
        }
        return dp[n-1][n/3];
    }
public:
    int maxSizeSlices(vector<int>& slices) {
        auto t(slices);
        t[0] = 0;
        slices.back() = 0;
        return max(maxWithoutCircle(t), maxWithoutCircle(slices));
    }
};
```

## 1391. Check if There is a Valid Path in a Grid

DFS，注意连通性，必须两个格子有同方向的路才算是连通。

```cpp
class Solution {
public:
    bool hasValidPath(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<bool>> v(m, vector<bool>(n, false));
        function<bool(int,int)> dfs = [&](int i, int j) {
            if (i == m - 1 && j == n - 1) return true;
            bool res = false;
            v[i][j] = true;
            if (grid[i][j] == 2) {
                if (i > 0 && !v[i-1][j] && (grid[i-1][j] == 2 
                    || grid[i-1][j] == 3 || grid[i-1][j] == 4)) res |= dfs(i-1, j);// up
                if (i < m-1 && !v[i+1][j] && (grid[i+1][j] == 2 
                    || grid[i+1][j] == 5 || grid[i+1][j] == 6)) res |= dfs(i+1, j);// down
            } else if (grid[i][j] == 1) {
                if (j > 0 && !v[i][j-1] && (grid[i][j-1] == 1 
                    || grid[i][j-1] == 4 || grid[i][j-1] == 6)) res |= dfs(i, j-1);// left
                if (j < n-1 && !v[i][j+1] && (grid[i][j+1] == 1 
                    || grid[i][j+1] == 3 || grid[i][j+1] == 5)) res |= dfs(i, j+1);// right
            } else if (grid[i][j] == 3) {
                if (j > 0 && !v[i][j-1] && (grid[i][j-1] == 1 
                    || grid[i][j-1] == 4 || grid[i][j-1] == 6)) res |= dfs(i, j-1);// left
                if (i < m-1 && !v[i+1][j] && (grid[i+1][j] == 2 
                    || grid[i+1][j] == 5 || grid[i+1][j] == 6)) res |= dfs(i+1, j);// down
            } else if (grid[i][j] == 4) {
                if (i < m-1 && !v[i+1][j] && (grid[i+1][j] == 2 
                    || grid[i+1][j] == 5 || grid[i+1][j] == 6)) res |= dfs(i+1, j);// down
                if (j < n-1 && !v[i][j+1] && (grid[i][j+1] == 1 
                    || grid[i][j+1] == 3 || grid[i][j+1] == 5)) res |= dfs(i, j+1);// right
            } else if (grid[i][j] == 5) {
                if (j > 0 && !v[i][j-1] && (grid[i][j-1] == 1 
                    || grid[i][j-1] == 4 || grid[i][j-1] == 6)) res |= dfs(i, j-1);// left
                if (i > 0 && !v[i-1][j] && (grid[i-1][j] == 2 
                    || grid[i-1][j] == 3 || grid[i-1][j] == 4)) res |= dfs(i-1, j);// up
            } else {
                if (i > 0 && !v[i-1][j] && (grid[i-1][j] == 2 
                    || grid[i-1][j] == 3 || grid[i-1][j] == 4)) res |= dfs(i-1, j);// up
                if (j < n-1 && !v[i][j+1] && (grid[i][j+1] == 1 
                    || grid[i][j+1] == 3 || grid[i][j+1] == 5)) res |= dfs(i, j+1);// right
            }
            return res;
        };
        return dfs(0, 0);
    }
};
```

## 1392. Longest Happy Prefix

KMP。

```cpp
class Solution {
public:
    string longestPrefix(string s) {
        int n = s.size();
        vector<int> next(s.size(), 0);
        int j = 0, i = 1;
        while (i < s.size()) {
            if (s[i] == s[j]) {
                next[i] = j + 1;
                i++; j++;
            } else if (j == 0) {
                i++;
            } else {
                j = next[j-1];
            }
        }
        return s.substr(0, next[n-1]);
    }
};
```

## 1397. Find All Good Strings

好难的一题，综合了深搜DP和KMP。

基本思想如下，假设当前准备填目标字符串`s`的第`i`位，有2种情况：

1. `s[0:i]`是s1的前缀，此时`s[i]`不能比`s1[i]`小，否则生成的目标串不在题目要求的区间内。
2. `s[0:i]`是s2的前缀，此时`s[i]`不能比`s2[i]`大。理由同上。

确定了能填的字符的范围，再考虑当前填下去是否有`evil`的前缀出现or增长。用一个参数`e`表示`s[0:i]`的后缀中等于`evil`的前缀的最长长度。那么当前填的字符`c`如果恰等于此前缀的后一位，则`e`加1，否则，使用KMP的前缀数组来寻找当前字符加入后`e`的新位置。考虑到题目中对数据大小的限制，可以简单的把参数做移位后用哈希表做缓存。

```cpp
class Solution {
public:
    int findGoodStrings(int n, string s1, string s2, string evil) {
        vector<int> next(evil.size(), 0);
        int i = 1, j = 0;
        while (i < evil.size()) {
            if (evil[i] == evil[j]) {
                next[i++] = ++j;
            } else if (j == 0) {
                next[i++] = 0;
            } else j = next[j-1];
        }
        unordered_map<int, int> mem;
        function<int(int,int,bool,bool)> dfs = [&](int i, int e, bool pre1, bool pre2) {
            if (e == evil.size()) return 0;
            if (i == n) return 1;
            int key = (i << 20) | (e << 8) | (pre1 << 1) | pre2;
            if (mem.find(key) != mem.end()) return mem[key];
            long res = 0;
            char start = pre1 ? s1[i] : 'a';
            char end = pre2 ? s2[i] : 'z';
            for (char c = start; c <= end; c++) {
                int e_ = e;
                while (c != evil[e_] && e_ > 0) e_ = next[e_-1];
                if (c == evil[e_]) {
                    res += dfs(i+1, e_+1, pre1&&s1[i]==c, pre2&&s2[i]==c);
                } else {
                    res += dfs(i+1, 0, pre1&&s1[i]==c, pre2&&s2[i]==c);
                }
            }
            return mem[key] = res % static_cast<long>(1e9+7);
        };
        return dfs(0, 0, true, true);
    }
};
```

## 1400. Construct K Palindrome Strings

数出现奇数次的子母个数。

```cpp
class Solution {
public:
    bool canConstruct(string s, int k) {
        vector<int> count(26, 0);
        for (auto c : s) count[c-'a']++;
        int oddCount = 0;
        for (auto i :count) if (i&1) oddCount++;
        return k <= s.size() && k >= oddCount;
    }
};
```

## 1401. Circle and Rectangle Overlapping

仔细分类。

```cpp
class Solution {
    inline int dis(int x, int y, int x0, int y0) {
        return (x-x0)*(x-x0) + (y-y0)*(y-y0);
    }
public:
    bool checkOverlap(int radius, int x_center, int y_center, int x1, int y1, int x2, int y2) {
        if (x_center >= x1 && y_center >= y1 && x_center <= x2 && y_center <= y2) return true;
        if (x_center < x1) {
            if (y_center < y1) return radius*radius >= dis(x_center, y_center, x1, y1);
            if (y_center > y2) return radius*radius >= dis(x_center, y_center, x1, y2);
            return x1 - x_center <= radius;
        } else if (x_center > x2) {
            if (y_center < y1) return radius*radius >= dis(x_center, y_center, x2, y1);
            if (y_center > y2) return radius*radius >= dis(x_center, y_center, x2, y2);
            return x_center - x2 <= radius;
        } else {
            if (y_center > y2) return y_center - y2 <= radius;
            else return y1 - y_center <= radius;
        }
        
    }
};
```

## 1402. Reducing Dishes

排序，贪心策略，不断地删最小的元素，直到总和非负。

```cpp
class Solution {
public:
    int maxSatisfaction(vector<int>& satisfaction) {
        sort(satisfaction.begin(), satisfaction.end());
        int sum = 0, res = 0, j = 0, k = 1;
        for (auto i : satisfaction) sum += i;
        while (sum < 0) sum -= satisfaction[j++];
        for (; j < satisfaction.size(); j++) res += satisfaction[j]*(k++);
        return res;
    }
};
```

## 1405. Longest Happy String

贪心地构造，每次选择剩余数量最多的，尝试加在后面，如果重复就把一个其他的子母作为间隔。

```cpp
class Solution {
public:
    string longestDiverseString(int a, int b, int c) {
        string res;
        priority_queue<pair<int, char>> pq;
        if (a > 0) pq.push({a, 'a'});
        if (b > 0) pq.push({b, 'b'});
        if (c > 0) pq.push({c, 'c'});
        while (!pq.empty()) {
            auto t = pq.top();
            pq.pop();
            if (!res.empty() && res.back() == t.second) {
                if (pq.empty()) break;
                auto tt = pq.top();
                pq.pop();
                res += tt.second;
                tt.first--;
                if (tt.first > 0) pq.push(tt);
            } 
            if (t.first == 1) {
                res += t.second;
                t.first--;
            } else {
                res.append(2, t.second);
                t.first -= 2;
            }
            if (t.first > 0) pq.push(t);
        }
        return res;
    }
};
```

## 1406. Stone Game III

这类博弈的问题，DP思路通常是玩家A从结点i开始能比B高几分。这样从另一个玩家的视角，实际上应该减去这个DP[i]的值。

```cpp
class Solution {
public:
    string stoneGameIII(vector<int>& s) {
        int n = stoneValue.size();
        vector<int> dp(n, INT_MIN);
        function<int(int)> dfs = [&](int i) {
            if (i == n) return 0;
            if (dp[i] != INT_MIN) return dp[i];
            if (i < n) dp[i] = max(dp[i], s[i]-dfs(i+1));
            if (i+1 < n) dp[i] = max(dp[i], s[i]+s[i+1]-dfs(i+2));
            if (i+2 < n) dp[i] = max(dp[i], s[i]+s[i+1]+s[i+2]-dfs(i+3));
            return dp[i];
        };
        int res = dfs(0);
        return res > 0 ? "Alice" : res == 0 ? "Tie" : "Bob"; 
    }
};
```

## 1411. Number of Ways to Paint N × 3 Grid

排列组合题，把上一层的涂色分成101型和012型。然后分别计数即可。

```CPP
class Solution {
public:
    int numOfWays(int n) {
        // 101 012
        // 6    6
        // 010 020 212 | 012 210
        // 101 121 | 120 201
        // 3a+2b 2a+2b
        int64_t a = 6, b = 6, res = a+b, M = 1000000007;
        while (--n>0) {
            res = 5*a + 4*b;
            tie(a, b) = make_tuple((3*a+2*b)%M, (2*a+2*b)%M);
        }
        return res%M;
    }
};
```



## 1423. Maximum Points You Can Obtain from Cards

```cpp
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        int n = cardPoints.size();
        for (int i = 1; i < n; i++) cardPoints[i] += cardPoints[i-1];
        if (n == k) return cardPoints.back();
        int res = cardPoints[n-1] - cardPoints[n-k-1];
        for (int i = 1; i <= k; i++) {
            res = max(res, cardPoints[i-1] + cardPoints[n-1] - cardPoints[n-k+i-1]);
        }
        return res;
    }
};
```

## 1442. Count Triplets That Can Form Two Arrays of Equal XOR

https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/discuss/624774/Rust-one-pass-O(n)-solution-0-ms

```cpp
class Solution {
public:
    int countTriplets(vector<int>& arr) {
        unordered_map<int, vector<int>> m;
        m[0].push_back(-1);
        int d = 0, res = 0;
        for (int i = 0; i < arr.size(); i++) {
            d ^= arr[i];
            if (m.find(d) != m.end()) for (auto t : m[d]) res += i-t-1;
            m[d].push_back(i);
        }
        return res;
    }
};
```

## 1452. People Whose List of Favorite Companies Is Not a Subset of Another List

如题，本来我是把每个字符串映射到一个包含该串的index集合，然后对每个index的字符串对应集合求交集；不过求交集效率比较低，耗时800ms；优化：由于总共只有不超过100个集合，因此用长度100的布尔串bitset，通过与运算求交集——因为底层运算进行了优化，所以速度降到了268ms；

```cpp
class Solution {
public:
    vector<int> peopleIndexes(vector<vector<string>>& favoriteCompanies) {
        int n = favoriteCompanies.size();
        vector<int> res;
        unordered_map<string, bitset<100>> m;
        for (int i = 0; i < n; i++) {
            for (auto &s : favoriteCompanies[i]) {
                m[s][i] = 1;
            }
        }
        for (int i = 0; i < n; i++) {
            bitset<100> pre;
            pre.set();
            for (auto &s : favoriteCompanies[i]) pre &= m[s];
            if (pre.count() == 1) res.push_back(i);
        }
        return res;
    }
};
```

## 1473. Paint House III

dp，考虑`dp[i][j][k]`为左侧颜色为i，从j开始涂色，目标领域数量为k的最小代价。其中i为0时表示左侧没有房子，递推公式很容易就能写出来，复杂度为$O(m^2 n^2)$。

```cpp
class Solution {
public:
    int minCost(vector<int>& houses, vector<vector<int>>& cost, int m, int n, int target) {
        int dp[21][100][101];
        for (auto& plane : dp) for (auto& row : plane) for (auto& x : row) x = -2;
        function<int(int,int,int)> dfs = [&](int i, int j, int k) {
            if (j == m) {
                if (k == 0) return 0;
                else return -1;
            }
            if (k < 0) return -1;
            if (dp[i][j][k] != -2) return dp[i][j][k];
            if (k > m - j) return dp[i][j][k] = -1;
            if (houses[j] != 0) {
                if (houses[j] == i)
                    return dp[i][j][k] = dfs(i, j+1, k);
                else
                    return dp[i][j][k] = dfs(houses[j], j+1, k-1);
            } else {
                int res = INT_MAX, t;
                if (i != 0) {
                    t = dfs(i, j+1, k);
                    if (t != -1) res = min(res, t + cost[j][i-1]);
                }
                if (k > 0) {
                    for (int r = 1; r <= n; r++) {
                        if (r != i) {
                            t = dfs(r, j+1, k-1);
                            if (t != -1) res = min(res, t + cost[j][r-1]);
                        }
                    }
                }
                if (res == INT_MAX) res = -1;
                return dp[i][j][k] = res;
            }
        };
        return dfs(0, 0, target);
    }
};
```

## 1531. String Compression II

这题好难，总是卡在这种隐晦的DP上。考虑`dp[i][m]`为从`s[i:]`中至多删除`m`个元素产生的最短压缩长度，此时如果`i`往后的长度不超过`k`，则全部删除即可，返回零。难点是下面如何正确缩小问题规模。我们总是尝试从`s[i:j]`中保留出现次数最多的元素，遍历`j`至底部。这样复杂度为$O(n^3)$

```cpp
class Solution {
    inline int dig(int t) {
        return t == 1 ? 0 : t < 10 ? 1 : t < 100 ? 2 : 3;
    }
public:
    int getLengthOfOptimalCompression(string s, int k) {
        vector<vector<int>> dp(s.size(), vector<int>(s.size(), -1));
        function<int(int,int)> dfs = [&](int i, int m)->int {
            if (m < 0) return s.size();
            if (i == s.size() || s.size() - i <= m) return 0;
            int &res = dp[i][m];
            if (res >= 0) return res;
            res = INT_MAX;
            int c[26] = {0};
            for (int j = i, most = 0; j < s.size(); j++) {
                most = max(most, ++c[s[j] - 'a']);
                res = min(res, dfs(j+1, m-(j-i+1-most)) + dig(most) + 1);
            }
            return res;
        };
        return dfs(0, k);
    }
};
```

## 1542. Find Longest Awesome Substring

寻找最长的子串，其至多有一个数出现奇数次，其余数出现偶数次。我们通过异或来记录每个位置出现的数的奇偶性，保存在mask中，那么mask最大为1024。用一张表来保存每个mask出现的最早位置，那么首先，相同mask之间的串一定所有数都出现偶数次；此外如果mask之间相差1个bit，那么中间的串就是出现了1个奇数次的数，也满足条件。这样整体时间复杂度为$O(10n)$。

```cpp
class Solution {
public:
    int longestAwesome(string s) {
        int mask = 0, res = 0;
        vector<int> dp(1024, 1e7);
        dp[0] = -1;
        for (int i = 0; i < s.size(); i++) {
            mask ^= (1 << (s[i] - '0'));
            res = max(res, i - dp[mask]);
            for (int j = 0; j <= 9; j++) {
                res = max(res, i - dp[mask ^ (1 << j)]);
            }
            dp[mask] = min(dp[mask], i);
        }
        return res;
    }
};
```

## 1557. Minimum Number of Vertices to Reach All Nodes

寻找入度为零的点即可。也可以按拓扑序遍历。

```cpp
class Solution {
public:
    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
        vector<int> res;
        vector<int> deg(n, 0);
        for (const auto &e : edges) deg[e[1]]++;
        for (int i = 0; i < n; i++) if (deg[i] == 0) res.push_back(i);
        return res;
    }
};
```

## 1567. Maximum Length of Subarray With Positive Product

贪心地选取，如果奇数个的话就尝试删去头部的负数。

```cpp
class Solution {
public:
    int getMaxLen(vector<int>& nums) {
        int last_0 = -1, res = 0, count = 0, last = -1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == 0) {
                last_0 = i; count = 0; last = -1;
            } else if (nums[i] < 0) {
                count++;
                if (last < 0) last = i;
            }
            if (count % 2 == 0) res = max(res, i-last_0);
            else res = max(res, i-last);
        }
        return res;
    }
};
```

## 1568. Minimum Number of Days to Disconnect Island

分三种情况：

* 如果本身就不连通，那直接返回0即可。
* 如果连通，存在割点，那么返回1.
* 否则返回2——这种情况必然存在右下角的方块，通过删除左上两块达到目的。

使用DFS求割点——Tarjan算法。

```cpp
class Solution {
public:
    int minDays(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> idx(m, vector<int>(n, -1));
        vector<vector<int>> low(m, vector<int>(n, INT_MAX));
        vector<vector<int>> dr({{0,1},{1,0},{0,-1},{-1,0}});
        int index = 0;
        bool foundCut = false;
        function<void(int,int,int,int)> dfs = [&](int i, int j, int fi, int fj) {
            idx[i][j] = index++;
            low[i][j] = idx[i][j];
            int count = 0;
            for (const auto &d : dr) {
                int ni = i + d[0], nj = j + d[1];
                if (ni == fi && nj == fj) continue;
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == 1) {
                    if (idx[ni][nj] == -1) {
                        dfs(ni, nj, i, j);
                        if (low[ni][nj] < low[i][j])
                            low[i][j] = low[ni][nj];
                        // 非根结点且存在孩子的low大于等于父亲的id，则是割点
                        if (fi != -1 && low[ni][nj] >= idx[i][j]) foundCut = true;
                        count++;
                    } else if (idx[ni][nj] < low[i][j])
                        low[i][j] = idx[ni][nj];
                }
            }// 根节点且孩子数量大于等于2，是割点
            if (fi == -1 && count >= 2) foundCut = true;
        };
        bool first = true;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && idx[i][j] == -1) {
                    if (!first) return 0;
                    dfs(i, j, -1, -1);
                    first = false;
                }
            }
        }
        return foundCut ? 1 : 2;
    }
};
```

## 1728. Cat and Mouse II

```cpp
class Solution {
public:
    bool canMouseWin(vector<string>& grid, int catJump, int mouseJump) {
        int xmax = grid.size(), ymax = grid[0].size();
        unordered_map<int, int> dp;
        function<bool(int,int,int,int,int)> dfs = [&](
            int mx, int my, int cx, int cy, int depth
        ) {
            int key = (mx<<28)|(my<<24)|(cx<<20)|(cy<<16)|depth;
            if (dp.find(key) != dp.end()) return dp[key];
            if (grid[mx][my] == 'F') return dp[key] = true;
            if (grid[cx][cy] == 'F') return dp[key] = false;
            if (mx == cx && my == cy) return dp[key] = false;
            if (depth > 67) return dp[key] = false;
            if (depth%2 == 0) { // mouse jump
                for (int gap = 1; gap <= mouseJump; gap++) {
                    if (mx + gap >= xmax || grid[mx+gap][my] == '#') break;
                    if (dfs(mx+gap, my, cx, cy, depth+1)) 
                        return dp[key] = true; // mouse win
                }
                for (int gap = 1; gap <= mouseJump; gap++) {
                    if (mx - gap < 0 || grid[mx-gap][my] == '#') break;
                    if (dfs(mx-gap, my, cx, cy, depth+1)) 
                        return dp[key] = true; // mouse win
                }
                for (int gap = 1; gap <= mouseJump; gap++) {
                    if (my + gap >= ymax || grid[mx][my+gap] == '#') break;
                    if (dfs(mx, my+gap, cx, cy, depth+1)) 
                        return dp[key] = true; // mouse win
                }
                for (int gap = 1; gap <= mouseJump; gap++) {
                    if (my - gap < 0 || grid[mx][my-gap] == '#') break;
                    if (dfs(mx, my-gap, cx, cy, depth+1)) 
                        return dp[key] = true; // mouse win
                }
                return dp[key] = false; // lose
            } else {// cat move
                for (int gap = 0; gap <= catJump; gap++) {
                    if (cx + gap >= xmax || grid[cx+gap][cy] == '#') break;
                    if (!dfs(mx, my, cx+gap, cy, depth+1))
                        return dp[key] = false;
                }
                for (int gap = 0; gap <= catJump; gap++) {
                    if (cx - gap < 0 || grid[cx-gap][cy] == '#') break;
                    if (!dfs(mx, my, cx-gap, cy, depth+1))
                        return dp[key] = false;
                }
                for (int gap = 0; gap <= catJump; gap++) {
                    if (cy + gap >= ymax || grid[cx][cy+gap] == '#') break;
                    if (!dfs(mx, my, cx, cy+gap, depth+1))
                        return dp[key] = false;
                }
                for (int gap = 0; gap <= catJump; gap++) {
                    if (cy - gap < 0 || grid[cx][cy-gap] == '#') break;
                    if (!dfs(mx, my, cx, cy-gap, depth+1))
                        return dp[key] = false;
                }
                return dp[key] = true;
            }
        };
        int x1, y1, x2, y2;
        for (int i= 0; i < xmax; i++) {
            for (int j = 0; j < ymax; j++) {
                if (grid[i][j] == 'M') {
                    x1 = i; y1 = j;
                }
                if (grid[i][j] == 'C') {
                    x2 = i; y2 = j;
                }
            }
        }
        return dfs(x1, y1, x2, y2, 0);
    }
};
```

