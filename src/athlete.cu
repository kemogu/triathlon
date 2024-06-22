#include "athlete.h"
#include <cstdlib>

Athlete::Athlete() : id(0), swim_speed(0), bike_speed(0), run_speed(0), position(0), total_position(0), total_time(0), current_parkur(0) {
    // default constructor
}

Athlete::Athlete(int athlete_id, float swim_speed_generated) : id(athlete_id), position(0), total_position(0), total_time(0), current_parkur(0) {

    swim_speed = swim_speed_generated;
    bike_speed = swim_speed * 3;
    run_speed = swim_speed / 3;
}

__device__ void Athlete::update_position(float speed, float time_seconds) {
    position += speed * time_seconds;
    total_time += time_seconds;

    // parkur geçişlerini kontrol etme
    switch (current_parkur) {
        case 0: // yüzme parkuru
            if (position >= 5000) {
                total_position = total_position + position;
                position = 0;
                switch_parkur();
            }
            break;
        case 1: // bisiklet parkuru
            if (position >= 40000) {
                total_position = total_position + position;
                position = 0;
                switch_parkur();
            }
            break;
        case 2: // koşu parkuru
            if (position >= 10000) {
                total_position = total_position + position;
                position = 0;
                current_parkur++;
            }
            break;
        case 3: // bitirilmiş parkur
            break;
    }
}

__device__ void Athlete::switch_parkur() {
    total_time += 10; // 10 saniye zaman kaybı
    current_parkur++;
}
