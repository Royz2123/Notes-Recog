import cv2
import os

import music_player

CROPPED_NOTES_PATH = "cropped_notes/"
DATABASE_NOTES_PATH = "database_notes/"

NOTE_LENGTH = 3
REST_LENGTH = 2

REST = "rest"

INNER_NOTE_REST = 0.3

# EXAMPLE MELODIES:
EXAMPLE_MELODY = ["g1", "e1", "e1", "rest", "f1", "d1", "d1", "rest", "c1", "d1", "e1", "f1", "g1", "g1", "g1"]
PLAYABLE_MELODY_2 = [
    {
        "type" : 0,
        "length" : 0.25,
        "freq" : 440,
    },
    {
        "type" : 0,
        "length" : 0.25,
        "freq" : 480,
    },
    {
        "type" : 0,
        "length" : 0.25,
        "freq" : 550,
    },
    {
        "type" : 0,
        "length" : 0.25,
        "freq" : 700,
    }
]

# musical constants
OCTAVE_LENGTH = 12
EMPTY_FREQ = 0

TYPES = {
    "note" : 0,
    "rest" : 1,
}
LENGTHS = {
    "sixteenth" : 0.0625,
    "eighth" : 0.125,
    "quarter" : 0.25,
    "half" : 0.5,
    "whole" : 1,
}
NUMS = {
    "c" : 1,
    "d" : 3,
    "e" : 5,
    "f" : 6,
    "g" : 8,
    "a" : 10,
    "b" : 12,
    "h" : 12,
}

# no point in trying to understand this... musical stuff
def calc_freq(note_name):
    note_num = (OCTAVE_LENGTH*int(note_name[1]) + NUMS[note_name[0]])
    return 440 * (2 **(float(note_num-22)/12))

def create_melody(extracted_notes, my_notes):
    melody = []
    for note in extracted_notes:
        melody.append(my_notes[note])
        melody.append({
            "type" : 1,
            "length" : INNER_NOTE_REST,
            "freq" : EMPTY_FREQ
        })
    return melody

def create_melody2(actual_notes):
    melody = []
    for note in actual_notes:
        freq = EMPTY_FREQ
        if note != "rest":
            freq = calc_freq(note)

        melody.append({
            "type" : 0,
            "length" : (1-INNER_NOTE_REST)*0.5,
            "freq" : freq
        })
        melody.append({
            "type" : 1,
            "length" : INNER_NOTE_REST*0.5,
            "freq" : EMPTY_FREQ
        })
    return melody

def main():
    data_notes = []
    # extract all filenames from the database
    for filename in os.listdir(DATABASE_NOTES_PATH):
        name_length = NOTE_LENGTH
        if REST in filename:
            name_length = REST_LENGTH

        # add to database
        if "other" not in filename:
            data_notes.append('-'.join(filename.split('-')[:name_length]))

    # sort lexicographically and remove clones
    data_notes = sorted(list(set(data_notes)))

    # create the database for notes:
    my_notes = {}
    for i in range(len(data_notes)):
        note = data_notes[i].split('-')
        my_notes[i] = {
            "type" : TYPES[note[0]],
            "length" : (1-INNER_NOTE_REST) * LENGTHS[note[1]]
        }
        if my_notes[i]["type"] == TYPES["rest"]:
            my_notes["freq"] = EMPTY_FREQ
        else:
            my_notes["freq"] = calc_freq("%s0" % note[2])


    # now we have database. call  neural_network for each note picture
    """
    extracted_notes = []
    for filename in sorted(os.listdir(CROPPED_NOTES_PATH)):
        extracted_notes.append(
            neural_network(
                cv2.imread(CROPPED_NOTES_PATH + filename)
            )
        )
    melody = create_melody(extracted_notes, my_notes)
    """

    melody = create_melody2(EXAMPLE_MELODY)

    # now extracted notes contains the entire melody
    # TODO: add "second hand"
    player = music_player.MusicPlayer(melody)
    player.add_melody()


if __name__ == "__main__":
    main()
