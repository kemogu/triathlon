#ifndef TRIATHLON_H
#define TRIATHLON_H

#include "athlete.h"
#include "team.h"
#include <algorithm>
#include <iostream>

class Triathlon {
public:
    Triathlon(int num_teams);
    ~Triathlon();

    void start_race();
    void print_results();
    void print_positions();

private:
    int num_teams;
    int num_athletes;
    Athlete* athletes;
    Team** teams;

    void initialize_teams();
    void update_positions(float time_interval);
    bool race_finished();
};

#endif // TRIATHLON_H
