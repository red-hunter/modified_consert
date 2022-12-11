from transformers import MarianMTModel, MarianTokenizer

class BackTranslate():

    def __init__(self,lang1, lang2):
        self.firstModelName = lang1
        self.secondModelName = lang2
        self.firstModeltkn = MarianTokenizer.from_pretrained(self.firstModelName)
        self.firstModel = MarianMTModel.from_pretrained(self.firstModelName)
        self.secondModeltkn = MarianTokenizer.from_pretrained(self.secondModelName)
        self.secondModel = MarianMTModel.from_pretrained(self.secondModelName)
        self.orig_text = []
        pass

    def run(self, text):
        translated_texts = self.perform_translation(text, self.firstModel, self.firstModeltkn)
        back_translated_texts = self.perform_translation(translated_texts, self.secondModel, self.secondModeltkn)
        return back_translated_texts

    def perform_translation(self, batch_texts, model, tokenizer):

        formated_batch_texts = batch_texts 
        translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]      
        return translated_texts

#'A group of boys in a yard is playing and a man is standing in the background'


#BTAug = BackTranslate('Helsinki-NLP/opus-mt-en-de','Helsinki-NLP/opus-mt-de-en')

#text = ["How Are You?"]

#translated_texts = BTAug.perform_translation(text, BTAug.firstModel, BTAug.firstModeltkn)
#print(translated_texts)    

#back_translated_texts = BTAug.perform_translation(translated_texts, BTAug.secondModel, BTAug.secondModeltkn)
#print(back_translated_texts)
 

