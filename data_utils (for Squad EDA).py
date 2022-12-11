import EDA as eda
import logging
from sentence_transformers import InputExample, LoggingHandler
import pandas as pd

n = 4
def load_sickr(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    if need_label:
        assert not no_pair, "Only paired texts need label"
    logging.info("Loading Squad (relatedness) dataset")
    all_samples = []
    
    squad_data = pd.read_pickle('all_squad_augmented.pkl')
    samples = []
    for _, data in squad_data.iterrows():
        sent1, sent2, label = data["sentence A"], data["sentence B"], 0
        if need_label:
            samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
        else:
            if no_pair:
                samples.append(InputExample(texts=[sent1]))
                samples.append(InputExample(texts=[sent2]))
            else:
                EDA_cases = random.sample(range(1,5),2)    
                sent1_BT = eda.Easy_Data_Augmentation(sent1,EDA_cases[0],n)
                sent1_BT = eda.Easy_Data_Augmentation(sent1_BT,EDA_cases[1],n)

                samples.append(InputExample(texts=[sent1, sent1_BT]))
    
    all_samples.extend(samples)
    logging.info(f"Loaded examples from Squad dataset, total {len(all_samples)} examples")
    return all_samples

result = load_sickr()
print(len(result))