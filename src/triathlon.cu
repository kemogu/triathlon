#include "triathlon.h"
#include <cuda_runtime.h>
#include <time.h> 

using namespace std; 

// ekran kartını kullanarak atlet posizyonunu güncelleme
__global__ void update_positions_kernel(Athlete* athletes, int num_athletes, float time_interval) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_athletes) {
        float speed;
        bool isFinished = false;
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
            case 3: // bitiş
                isFinished = true;
                break;
        }
        if (isFinished == false)
        {
            athletes[idx].update_position(speed, time_interval);
        }
        
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
        athletes[i*3] = Athlete(i*3, rand()%5+1);
        athletes[i*3+1] = Athlete(i*3+1, rand()%5+1);
        athletes[i*3+2] = Athlete(i*3+2, rand()%5+1);
        teams[i] = new Team(i, &athletes[i*3], &athletes[i*3+1], &athletes[i*3+2]);
    }
}

void Triathlon::update_positions(float time_interval) {
    Athlete* d_athletes;
    cudaMalloc(&d_athletes, num_athletes * sizeof(Athlete));
    cudaMemcpy(d_athletes, athletes, num_athletes * sizeof(Athlete), cudaMemcpyHostToDevice);

    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, update_positions_kernel, 0, num_athletes);

    gridSize = (num_athletes + blockSize - 1) / blockSize;
    
    update_positions_kernel<<<gridSize, blockSize>>>(d_athletes, num_athletes, time_interval);

    cudaMemcpy(athletes, d_athletes, num_athletes * sizeof(Athlete), cudaMemcpyDeviceToHost);
    cudaFree(d_athletes);
}

bool Triathlon::race_finished() {
    int control_of_num_athletes = 0;
    for (int i = 0; i < num_athletes; ++i) {
        if (athletes[i].current_parkur == 3) {
            // 55 km toplam parkur ve koşu parkuru bitiş koşulları
            control_of_num_athletes ++;
        }
    }
    if(num_athletes == control_of_num_athletes)
        return true;
    return false;
}

void Triathlon::start_race() {
    const float time_interval = 1.0f; // saniyede bir güncelleme
    while (!race_finished()) {
        update_positions(time_interval);
        print_positions();
    }
    //print_positions();
    print_results();
}
void Triathlon::print_positions() {
    for (int i = 0; i < num_athletes; ++i) {
        std::cout << "Athlete " << athletes[i].id << ": " << athletes[i].position <<" ," << athletes[i].total_position  << " meters, Total time: " << athletes[i].total_time << " seconds" << " state: " << athletes[i].current_parkur << std::endl;
    }
}

void Triathlon::print_results() {
    Athlete* first_athlete = find_first_athlete();
    Team* best_team = find_best_team();

    if (first_athlete) {
        std::cout << "First athlete to finish: Athlete " << first_athlete->id << " with total time: " << first_athlete->total_time << " seconds" << std::endl;
    }

    if (best_team) {
        std::cout << "Best team: Team " << best_team->team_id << " with total time: " << best_team->team_total_time << " seconds" << std::endl;
    }
    for (int i = 0; i < num_teams; ++i) {
        teams[i]->calculate_team_total_time();
        std::cout << "Team Name: Team " << teams[i]->team_id << " with total time: " << teams[i]->team_total_time << " seconds" << std::endl;
    }
}
Athlete* Triathlon::find_first_athlete() {
    Athlete* first_athlete = nullptr;
    float min_time = std::numeric_limits<float>::max();

    for (int i = 0; i < num_athletes; ++i) {
        if (athletes[i].current_parkur == 3 && athletes[i].total_time < min_time) {
            min_time = athletes[i].total_time;
            first_athlete = &athletes[i];
        }
    }

    return first_athlete;
}

Team* Triathlon::find_best_team() {
    Team* best_team = nullptr;
    float min_time = std::numeric_limits<float>::max();

    for (int i = 0; i < num_teams; ++i) {
        teams[i]->calculate_team_total_time();
        float team_time = teams[i]->team_total_time;
        if (team_time < min_time) {
            min_time = team_time;
            best_team = teams[i];
        }
    }

    return best_team;
}

