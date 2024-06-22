#ifndef ATHLETE_H
#define ATHLETE_H

class Athlete {
public:
    int id;
    float swim_speed;
    float bike_speed;
    float run_speed;
    float position;
    float total_position;
    float total_time;
    int current_parkur; // 0 yüzme, 1 bisiklet, 2 koşma , 3 bitiş

    Athlete();  // default constructor
    Athlete(int athlete_id, float swim_speed_generated);  // parametreli constructor

    __device__ void update_position(float speed, float time_seconds);
    __device__ void switch_parkur(); // parkur geçişi fonksiyonu
};

#endif // ATHLETE_H
