#include "triathlon.h"

int main() {
    srand(time(0)); // her çalışmada rastgele hızlar üretmek için 
    const int num_teams = 3;
    Triathlon triathlon(num_teams);
    triathlon.start_race();
    return 0;
}
