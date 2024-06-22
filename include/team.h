#ifndef TEAM_H
#define TEAM_H

#include "athlete.h"

class Team {
public:
    int team_id;
    Athlete* athletes[3];
    float team_total_time;

    Team(int id, Athlete* a1, Athlete* a2, Athlete* a3);
    void calculate_team_total_time();
    Athlete* get_athlete(int index);
};

#endif
