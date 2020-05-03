from pydub import AudioSegment

sound1 = AudioSegment.from_file("bonus/my_mix/s1.wav")
sound2 = AudioSegment.from_file("bonus/my_mix/s2.wav")

combined = sound1.overlay(sound2)

combined.export("bonus/my_mix/my_audio.wav", format='wav')
