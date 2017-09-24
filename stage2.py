import cv2
import os

import music_player

CROPPED_NOTES_PATH = "/cropped_notes"
DATABASE_NOTES = "/database_notes"

NOTE_LENGTH = 3
REST_LENGTH = 2

REST = "rest"

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

OCTAVE_LENGTH = 12

NUMS = {
    "c" : 1,
    "d" : 3,
    "e" : 5,
    "f" : 6,
    "g" : 8,
    "a" : 10,
    "b" : 12,
}

def calc_freq(note_name):
    note_num = OCTAVE_LENGTH*int(note_num[1]) + NUMS[note_num[0]]
    return 440 * (2 **(note_num/12))

def main():
    data_notes = []
    # extract all filenames from the database
    for filename in os.listdir(CROPPED_NOTES_PATH):
        name_length = (REST in filename) ? REST_LENGTH : NOTE_LENGTH
        data_notes.append('-'.join(filename.split('-')[:name_length]))

    # sort lexicographically and remove clones
    data_notes = sorted(list(set(data_notes)))

    # create the database for notes:
    my_notes = {}
    for i in range(len(data_notes)):
        note = data_notes[i].split('-')
        my_notes[i] = {
            "type" : TYPES[note[0]],
            "length" : LENGTH[note[1]]
        }
        if len(note) == NOTE_LENGTH:
            my_notes["freq"] = calc_freq(note[2])

    # now we have database. call  neural_network for each note picture
    extracted_notes = []
    for filename in os.listdir(CROPPED_NOTES_PATH):
        extracted_notes.append(
            neural_network(
                cv2.imread(CROPPED_NOTES_PATH + filename)
            )
        )

    # create melody with objects from note database
    melody = []
    for note in extracted_notes:
        melody.append(my_notes[note])

    # now extracted notes contains the entire melody
    # TODO: add "second hand"
    player = music_player.MusicPlayer(melody)
    player.add_melody()


if __name__ == "__main__":
    main()
