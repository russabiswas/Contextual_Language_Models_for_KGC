from __future__ import absolute_import, division, print_function
import argparse, csv, logging, os, random, sys
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()
        
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        return ["0", "1"]

    def get_entities(self, data_dir):
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):
            
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":
                triple_label = line[3]
                if triple_label == "1":
                    label = "1"
                else:
                    label = "0"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    tmp_head = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                        if tmp_triple_str not in lines_str_set:
                            break                    
                    tmp_head_text = ent2text[tmp_head]
                    examples.append(
                        InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    args = parser.parse_args()

    processors = {"kg": KGProcessor}

    task_name = args.task_name.lower()
    
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels(args.data_dir)	
    entity_list = processor.get_entities(args.data_dir)

    test_examples = None
    test_examples = processor.get_test_examples(args.data_dir)
    num_test_examples = len(test_examples)

    for i in range(num_test_examples):
        print(test_examples[i].text_a, '\t', test_examples[i].text_b, '\t', test_examples[i].text_c, '\t', test_examples[i].label)
		
if __name__ == "__main__":
    main()