import whisper
import speech_recognition as sr
import base64 

from pydub import AudioSegment


class SpeechToTextConverter():
    def __init__(self):
        self.model = whisper.load_model("base")


    def processAudio(self):
        _a = whisper.load_audio(self.AudioFileName)
        _a = whisper.pad_or_trim(_a)

        _m = whisper.log_mel_spectrogram(_a).to(self.model.device)

        _options = whisper.DecodingOptions(fp16 = False)
        _o = whisper.decode(self.model, _m, _options)

        transcription = _o.text 
        return transcription
    
    def convertFileToWav(self):
        sound = AudioSegment.from_file(self.AudioFileName, format='m4a')
        export_name = "Output_Convert.wav"
        file_handle = sound.export(export_name, format='wav')
        self.AudioFileName = export_name
    
    def transform(self, AudioFileName):
        self.AudioFileName = AudioFileName
        self.convertFileToWav()        
        result = self.processAudio()

        return result


