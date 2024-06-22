#include "triathlon.h"
#include <cuda_runtime.h>

// ekran kartını kullanarak atlet posizyonunu güncelleme
__global__ void update_positions_kernel(Athlete* athletes, int num_athletes, float time_interval) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_athletes) {
        float speed;
        switch (athletes[idx].current_parkur) {
            case 0: // yüzme parkuru
                speed = athletes[idx].swim_speed;
                break;
            case 1: // bisiklet parkuru
                speed = athletes[idx].bike_speed;
                break;
            case 2: // koşu parkuru
                speed = athletes[idx].run_speed;
                break;
        }
        athletes[idx].update_position(speed, time_interval);
    }
}

// parametreli constructer
Triathlon::Triathlon(int num_teams)
    : num_teams(num_teams), num_athletes(num_teams * 3) {
    athletes = new Athlete[num_athletes];
    teams = new Team*[num_teams];
    initialize_teams();
    
}

Triathlon::~Triathlon() {
    delete[] athletes;
    for (int i = 0; i < num_teams; ++i) {
        delete teams[i];
    }
    delete[] teams;
}

void Triathlon::initialize_teams() {
    for (int i = 0; i < num_teams; ++i) {
        athletes[i*3] = Athlete(i*3);
        athletes[i*3+1] = Athlete(i*3+1);
        athletes[i*3+2] = Athlete(i*3+2);
        teams[i] = new Team(i, &athletes[i*3], &athletes[i*3+1], &athletes[i*3+2]);
    }
}

void Triathlon::update_positions(float time_interval) {
    Athlete* d_athletes;
    cudaMalloc(&d_athletes, num_athletes * sizeof(Athlete));
    cudaMemcpy(d_athletes, athletes, num_athletes * sizeof(Athlete), cudaMemcpyHostToDevice);

    update_positions_kernel<<<(num_athletes + 255) / 256, 256>>>(d_athletes, num_athletes, time_interval);

    cudaMemcpy(athletes, d_athletes, num_athletes * sizeof(Athlete), cudaMemcpyDeviceToHost);
    cudaFree(d_athletes);
}

bool Triathlon::race_finished() {
    for (int i = 0; i < num_athletes; ++i) {
        if (athletes[i].current_parkur == 2 && athletes[i].position >= 55000) {
            // 55 km toplam parkur ve koşu parkuru bitiş koşulları
            return true;
        }
    }
    return false;
}

void Triathlon::start_race() {
    const float time_interval = 1.0f; // saniyede bir güncelleme
    while (!race_finished()) {
        update_positions(time_interval);
    }
    print_positions();
}

void Triathlon::print_results() {
    for (int i = 0; i < num_teams; ++i) {
        teams[i]->calculate_team_total_time();
    }

    std::sort(teams, teams + num_teams, [](Team* a, Team* b) {
        return a->team_total_time < b->team_total_time;
    });

    for (int i = 0; i < num_teams; ++i) {
        std::cout << "Team " << teams[i]->team_id << ": " << teams[i]->team_total_time << " seconds" << std::endl;
    }
}

void Triathlon::print_positions() {
    for (int i = 0; i < num_athletes; ++i) {
        std::cout << "Athlete " << athletes[i].id << ": " << athletes[i].position << " meters, Total time: " << athletes[i].total_time << " seconds" << std::endl;
    }
}
