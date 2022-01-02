class Solution
{
    // 使用log去计算最终的乘积的数字位数（实际上是算它是x.xxx * 10^y）
public:
    string abbreviateProduct(int left, int right)
    {
        int z = 0, c2 = 0, c5 = 0;
        long double logsum = 0;
        for (int n = left; n <= right; n++)
        {
            logsum += log10(n);
            int t = n;
            while (t > 0 && t % 2 == 0)
            {
                c2++;
                t /= 2;
            }
            while (t > 0 && t % 5 == 0)
            {
                c5++;
                t /= 5;
            }
        }
        z = min(c2, c5);
        c2 = z;
        c5 = z;
        int numdigits = logsum;
        if (numdigits - z <= 10)
        {
            int64_t last = 1;
            for (int n = left; n <= right; n++)
            {
                last *= n;
                while (last % 10 == 0)
                    last /= 10;
            }
            return to_string(last) + "e" + to_string(z);
        }
        long double y = logsum - (long double)(int)logsum;
        int first = pow(10, y + 4);
        int64_t last = 1;
        for (int n = left; n <= right; n++)
        {
            last *= n;
            while (last % 2 == 0 && c2 > 0)
            {
                last /= 2;
                c2--;
            }
            while (last % 5 == 0 && c5 > 0)
            {
                last /= 5;
                c5--;
            }
            last %= 100000;
        }
        char res[50];
        sprintf(res, "%d...%05de%d", first, last, z);
        return res;
    }
};