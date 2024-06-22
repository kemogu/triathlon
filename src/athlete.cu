#include "athlete.h"
#include <cstdlib>

Athlete::Athlete() : id(0), swim_speed(0), bike_speed(0), run_speed(0), position(0), total_time(0), current_parkur(0) {
    // default constructor
}

Athlete::Athlete(int athlete_id) : id(athlete_id), position(0), total_time(0), current_parkur(0) {
    swim_speed = std::rand() % 5 + 1;
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
                switch_parkur();
            }
            break;
        case 1: // bisiklet parkuru
            if (position >= 45000) {
                switch_parkur();
            }
            break;
        case 2: // koşu parkuru
            // koşu parkurunda konumu kontrol etmeye gerek yok, yarış bittikten sonra sadece zaman toplanacak
            break;
    }
}

__device__ void Athlete::switch_parkur() {
    total_time += 10; // 10 saniye zaman kaybı
    current_parkur++;
}
