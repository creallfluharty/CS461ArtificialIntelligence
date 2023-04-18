from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import count
from typing import (
    List,
    Optional,
    Set,
    Dict,
)
import json
import random

from scipy.special import softmax
from numpy.random import choice


@dataclass(frozen=True)
class Room:
    building: str
    number: int
    capacity: int


@dataclass
class Activity:
    subject: str
    course_number: int
    section: Optional[str]
    expected_enrollment: int
    preferred_facilitators: Set[str]
    other_facilitators: Set[str]

    def get_id(self):
        return f"{self.subject}{self.course_number:03}{self.section or ''}"


@dataclass
class Schedule:
    activities: List[Activity]
    facilitators: List[str]
    rooms: List[Room]
    times: List[int]


@dataclass
class ActivityAssignment:
    activity: Activity
    facilitator: str
    room: Room
    time: int

    @staticmethod
    def make_random_assignment(schedule: Schedule, activity: Activity) -> 'ActivityAssignment':
        return ActivityAssignment(
            activity,
            random.choice(schedule.facilitators),
            random.choice(schedule.rooms),
            random.choice(schedule.times),
        )


@dataclass
class ScheduleAssignment:
    schedule: Schedule
    activities: Dict[str, ActivityAssignment]

    @staticmethod
    def make_random_assignment(schedule: Schedule) -> 'ScheduleAssignment':
        activities = {
            activity.get_id(): ActivityAssignment.make_random_assignment(schedule, activity)
            for activity in schedule.activities
        }
        return ScheduleAssignment(schedule, activities)


def get_fitness(schedule: ScheduleAssignment):
    fitness = 0

    room_time_load = Counter()
    facilitator_time_load = Counter()
    facilitator_time_building_schedule = defaultdict(set)
    facilitator_load = defaultdict(lambda: 0)

    building_clusters = {
        'Roman': 2,
        'Beach': 2,
        'Slater': 1,
        'Loft': 1,
        'Logos': 1,
        'Frank': 1,
    }

    # activity fitness
    for activity in schedule.activities.values():
        room_time_load[activity.room, activity.time] += 1
        facilitator_load[activity.facilitator] += 1
        facilitator_time_load[activity.facilitator, activity.time] += 1
        facilitator_time_building_schedule[activity.facilitator, activity.time].add(building_clusters[activity.room.building])

        capacity_ratio = activity.room.capacity / activity.activity.expected_enrollment
        if capacity_ratio < 1:
            fitness -= 0.5
        elif capacity_ratio > 3:
            fitness -= 0.2
        elif capacity_ratio > 6:
            fitness -= 0.4
        else:
            fitness += 0.3

        if activity.facilitator in activity.activity.preferred_facilitators:
            fitness += 0.5
        elif activity.facilitator in activity.activity.other_facilitators:
            fitness += 0.2
        else:
            fitness -= 0.1

    # penalize room/time conflicts
    for load in room_time_load.values():
        if load >= 2:
            fitness -= 0.5 * load

    # penalize schedules that give professors too many or too few activities
    for facilitator, load in facilitator_load.items():
        if facilitator == 'Tyler' and load < 2:
            continue

        if load in (1, 2):  # apparently not 0 though
            fitness -= 0.4
        elif load > 4:
            fitness -= 0.5

    # penalize (very lightly, apparently) schedules that require facilitators to be in multiple places at once
    for load in facilitator_time_load.values():
        if load > 1:  # apparently this is a one-time thing?
            fitness -= 0.2

    # penalize schedules that require the instructor to travel far between consecutive activities, otherwise reward
    for (facilitator, time), building_groups in facilitator_time_building_schedule.items():
        for building_group in building_groups:  # Still O(num_activities)
            prev = facilitator_time_building_schedule.get((facilitator, time - 1), tuple())
            if len(prev) > 1 or (len(prev) == 1 and building_group not in prev):
                fitness -= 0.4
            else:
                fitness += 0.5

    # "activity-specific adjustments"
    sla101_gap = abs(schedule.activities['SLA101A'].time - schedule.activities['SLA101B'].time)
    if sla101_gap > 4:
        fitness += 0.5
    elif sla101_gap == 0:
        fitness -= 0.5

    sla191_gap = abs(schedule.activities['SLA191A'].time - schedule.activities['SLA191B'].time)
    if sla191_gap > 4:
        fitness += 0.5
    elif sla191_gap == 0:
        fitness -= 0.5

    for sla101 in (schedule.activities['SLA101A'], schedule.activities['SLA101B']):
        for sla191 in (schedule.activities['SLA191A'], schedule.activities['SLA191B']):
            gap = abs(sla101.time - sla191.time)
            if gap == 1:
                if building_clusters[sla101.room.building] == building_clusters[sla191.room.building]:
                    fitness += 0.5
                else:
                    fitness -= 0.4
            elif gap == 2:
                fitness += 0.25
            elif gap == 0:
                fitness -= 0.25

    return fitness


def cross_schedules(schedule: Schedule, a: ScheduleAssignment, b: ScheduleAssignment, mutation_rate):
    chiasma = random.randint(0, len(schedule.activities) - 1)
    new_activities = {}
    for i, activity in enumerate(schedule.activities):
        activity_name = activity.get_id()
        new_activities[activity_name] = (
            ActivityAssignment.make_random_assignment(schedule, activity) if random.random() < mutation_rate
            else a.activities[activity_name] if i < chiasma
            else b.activities[activity_name]
        )

    return ScheduleAssignment(a.schedule, new_activities)


def get_next_generation(schedule, population, population_fitness, parent_pool_size, mutation_rate):
    softmax_pop_fitness = softmax(population_fitness)
    parent_population = choice(population, parent_pool_size, replace=False, p=softmax_pop_fitness)
    new_generation = []
    for _ in population:
        a, b = choice(parent_population, 2, replace=False)
        new_generation.append(cross_schedules(schedule, a, b, mutation_rate))

    return new_generation


def print_schedule(schedule: ScheduleAssignment):
    time_slots = {time: [] for time in schedule.schedule.times}

    for activity in schedule.activities.values():
        time_slots[activity.time].append(activity)

    for time, activities in time_slots.items():
        print(f"{time: <2}:00", end=' | ')
        print(
            " | ".join(
                f"{activity.activity.get_id(): <7} "
                f"{activity.facilitator: <8} "
                f"{activity.room.building: <5} "
                f"{activity.room.number:0<3}"
                for activity in activities
            )
        )


def main():
    with open("activity_info.json") as f:
        activity_info = json.load(f)

    activities = [
        Activity(
            activity['subject'],
            activity['course number'],
            activity['section'],
            activity['expected enrollment'],
            set(activity['preferred facilitators']),
            set(activity['other facilitators']),
        )
        for activity in activity_info["activities"]
    ]

    facilitators = activity_info['facilitators']

    rooms = [
        Room(
            room['building'],
            room['number'],
            room['capacity'],
        )
        for room in activity_info['rooms']
    ]

    times = activity_info['times']

    schedule = Schedule(
        activities,
        facilitators,
        rooms,
        times,
    )

    POPULATION_SIZE = 500
    PARENT_POPULATION_SIZE = int(POPULATION_SIZE * 0.2)
    MUTATION_RATE_DECAY = 0.5
    mutation_rate = 0.1

    population = [ScheduleAssignment.make_random_assignment(schedule) for _ in range(POPULATION_SIZE)]
    population_fitness = [get_fitness(s) for s in population]

    for g in count():
        avg_prev_population_fitness = sum(population_fitness) / len(population)
        population = get_next_generation(schedule, population, population_fitness, PARENT_POPULATION_SIZE, mutation_rate)
        population_fitness = [get_fitness(s) for s in population]
        avg_population_fitness = sum(population_fitness) / len(population_fitness)

        print(f"generation {g} has average fitness {avg_prev_population_fitness}")

        if g > 100:
            if avg_population_fitness / avg_prev_population_fitness < 1.01:
                break

            mutation_rate *= MUTATION_RATE_DECAY

    best_schedule_ix, best_fitness = max(enumerate(population_fitness), key=lambda p: p[1])
    print(f"Best Schedule with fitness {best_fitness} was:")
    print_schedule(population[best_schedule_ix])


if __name__ == '__main__':
    main()
