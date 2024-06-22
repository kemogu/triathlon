#include "team.h"

Team::Team(int id, Athlete* a1, Athlete* a2, Athlete* a3) : team_id(id), team_total_time(0) {
    athletes[0] = a1;
    athletes[1] = a2;
    athletes[2] = a3;
}

void Team::calculate_team_total_time() {
    team_total_time = athletes[0]->total_time + athletes[1]->total_time + athletes[2]->total_time;
}
Athlete* Team::get_athlete(int index) {
    if (index >= 0 && index < 3) {
        return athletes[index];
    }
    return nullptr;
}
