from sentence_transformers import InputExample, LoggingHandler
import logging

def load_sickr(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    if need_label:
        assert not no_pair, "Only paired texts need label"
    logging.info("Loading SICK (relatedness) dataset")
    all_samples = []
    if use_all_unsupervised_texts:
        splits = ["train", "trial", "test_annotated"]
    else:
        splits = ["test_annotated"]
    for split in splits:
        sick_data_path = f"/home/niloofarhp/Documents/Projects/ConSERT/data/downstream/SICK/SICK_{split}.txt"
        with open(sick_data_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        samples = []
        for line in lines[1:]:
            _, sent1, sent2, label, _ = line.split("\t")
            if need_label:
                samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
            else:
                if no_pair:
                    samples.append(InputExample(texts=[sent1]))
                    samples.append(InputExample(texts=[sent2]))
                else:
                    samples.append(InputExample(texts=[sent1, sent2]))
        all_samples.extend(samples)
    logging.info(f"Loaded examples from SICK dataset, total {len(all_samples)} examples")
    return all_samples

#to load dataset STS completely
result = load_sickr()
print(len(result))

