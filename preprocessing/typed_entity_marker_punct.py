import pandas as pd
import copy

class preprocessing_dataset_TypedEntityMarker:
    """
    기존의 DataFrame의 sentence 열에 Typed Entity Marker를 부착하는 함수 클래스입니다.
    attach_TypedEntityMarker 함수를 통해 사용할 수 있습니다.
    """
    def __init__(self):
        pass

    def insert_word(self, text, start_idx, word):
        """
        기존의 text 사이에 word를 추가합니다.
        Args:
            text: 변경할 text입니다.
            start_idx: word를 넣고자 하는 위치입니다.
            word: 넣고자 하는 단어입니다.
        """
        return text[:start_idx] + word + text[start_idx:]

    def insert_token(self, sentence, insert_word, subject_start, subject_end, subject_type, object_start, object_end, object_type):
        """
        sentence에 subject와 object word 전후에 typed entity marker를 추가합니다.
        """
        subject_end = int(subject_end)
        object_end = int(object_end)
        subject_start = int(subject_start)
        object_start = int(object_start)
        
        if subject_end >= object_end:
            sentence = self.insert_word(sentence, subject_end+1, f"</S:{subject_type}>")
            sentence = self.insert_word(sentence, subject_start, f"<S:{subject_type}>")
            sentence = self.insert_word(sentence, object_end+1, f"</O:{object_type}>")
            sentence = self.insert_word(sentence, object_start, f"<O:{object_type}>")
        else:
            sentence = self.insert_word(sentence, object_end+1, f"</O:{object_type}>")
            sentence = self.insert_word(sentence, object_start, f"<O:{object_type}>")
            sentence = self.insert_word(sentence, subject_end+1, f"</S:{subject_type}>")
            sentence = self.insert_word(sentence, subject_start, f"<S:{subject_type}>")
        
        # @ * person * Bill @ was born in # ∧ city ∧ Seattle #.
        return sentence
    
    def insert_token_punct(self, sentence, insert_word, subject_start, subject_end, subject_type, object_start, object_end, object_type):
        """
        sentence에 subject와 object word 전후에 typed entity marker punct를 추가합니다.
        """
        subject_end = int(subject_end)
        object_end = int(object_end)
        subject_start = int(subject_start)
        object_start = int(object_start)
        
        if subject_end >= object_end:
            sentence = self.insert_word(sentence, subject_end+1, f" @")
            sentence = self.insert_word(sentence, subject_start, f"@ * {subject_type} * ")
            sentence = self.insert_word(sentence, object_end+1, f" #")
            sentence = self.insert_word(sentence, object_start, f"# ^ {object_type} ^ ")
        else:
            sentence = self.insert_word(sentence, object_end+1, f" #")
            sentence = self.insert_word(sentence, object_start, f"# ^ {object_type} ^ ")
            sentence = self.insert_word(sentence, subject_end+1, f" @")
            sentence = self.insert_word(sentence, subject_start, f"@ * {subject_type} * ")
        
        # @ * person * Bill @ was born in # ∧ city ∧ Seattle #.
        return sentence
    
    def insert_source(self, sentence, source):
        sentence = self.insert_word(sentence, 0, f"< {source} > ")
        
        return sentence

    def attach_TypedEntityMarker(self, dataset, punct=True):
        """
        기존의 DataFrame의 sentence 열에 Typed Entity Marker를 부착합니다.
        Args:
            dataset: 적용할 DataFrame입니다.
            punct: punct를 적용할 것인지의 여부입니다. False이면 Typed_Entity_Marker를 적용합니다.
        
        Example:
            pdt = preprocessing_dataset_TypedEntityMarker()
            train_df = pdt.attach_TypedEntityMarker(train_df)
        """
        
        out_dataset = copy.deepcopy(dataset)
        
        if punct:
            funct = self.insert_token_punct
        else:
            funct = self.insert_token
            
        sentences = []
        sbj_dicts = []
        obj_dicts = []
        src_dicts = []
        for idx, d in out_dataset.iterrows():
            sentences.append(d.sentence)
            sbj_dicts.append(eval(d.subject_entity))
            obj_dicts.append(eval(d.object_entity))
            src_dicts.append(d.source)
        
        for idx, sentence in enumerate(sentences):
            sbj_dict = sbj_dicts[idx]
            obj_dict = obj_dicts[idx]
            src_dict = src_dicts[idx]
            sentence = funct(sentence, self.insert_word, sbj_dict['start_idx'], sbj_dict['end_idx'], sbj_dict['type'], 
                        obj_dict['start_idx'], obj_dict['end_idx'], obj_dict['type'])
            sentence = self.insert_source(sentence, src_dict)
            sentences[idx] = sentence
        
        out_dataset['sentence'] = sentences

        return out_dataset