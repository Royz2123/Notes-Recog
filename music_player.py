import wave, struct, math

class MusicPlayer(object):
    def __init__(self, melody):
        self._sampleRate = 44100.0 # hertz
        self._melody = melody

        self._wavef = wave.open('sound.wav','w')
        self._wavef.setnchannels(1) # mono
        self._wavef.setsampwidth(2)
        self._wavef.setframerate(self._sampleRate)

    def add_note(self, freq, duration):
        for i in range(int(duration * self._sampleRate)):
            value = int(32767.0*math.cos(freq*math.pi*float(i)/float(self._sampleRate)))
            data = struct.pack('<h', value)
            self._wavef.writeframesraw( data )

    def add_melody(self):
        for note in self._melody:
            self.add_note(note["freq"], note["length"])
        self._wavef.writeframes('')
        self._wavef.close()
