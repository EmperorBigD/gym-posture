"""All supported exercises, by CLI name."""

from gym_posture.exercises.base import Exercise
from gym_posture.exercises.dumbbell_press import DumbbellPress
from gym_posture.exercises.squat import Squat

EXERCISES: dict[str, Exercise] = {
    "squat": Squat(),
    "press": DumbbellPress(),
}
