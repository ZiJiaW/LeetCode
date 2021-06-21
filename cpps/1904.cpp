class Solution {
public:
    int numberOfRounds(string startTime, string finishTime) {
        auto startH = stoi(startTime.substr(0, 2));
        auto startM = stoi(startTime.substr(3, 2));
        auto endH = stoi(finishTime.substr(0, 2));
        auto endM = stoi(finishTime.substr(3, 2));
        if (startM % 15 != 0) startM = 15 * (startM / 15 + 1);
        if (endM % 15 != 0) endM = 15 * (endM / 15);
        if (endH < startH || (endH == startH && endM < startM)) endH += 24;
        return (endH*60+endM - startH*60-startM) / 15;
    }
};