# pylint: disable=E1101
import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn

import random
from pathlib import Path
import re

from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
from difflib import ndiff
from venn import venn

import torch

from .custom_classes import Vocabulary, DatasetObject


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"] == sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]
                                     == sentence_id]

        def decode_word(x): return self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,
                                                   "token_id"].apply(decode_word)

        sample = " "
        for i, t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True

            if not is_ner:
                sample += t_row["token"] + " "

        # print(self.files_pathes[sentence_id])
        return sample.rstrip()



class DataLoader(DataLoaderBase):

    def __init__(self, data_dir: str, device=None):
        random.seed(42)

        global pad_token
        global start_sent_token
        global end_sent_token
        pad_token = "padpad"
        start_sent_token = "startseq"
        end_sent_token = "endseq"
        unkown_token = "unk"

        self.word_vocab = Vocabulary()
        self.ner_vocab = Vocabulary()
        self.max_sample_length = 0

        self.word_vocab.add(pad_token)
        self.word_vocab.add(start_sent_token)
        self.word_vocab.add(end_sent_token)
        self.word_vocab.add(unkown_token)

        self.ner_vocab.add(pad_token)
        self.ner_vocab.add("O")

        # needed for troubleshooting
        self.files_pathes = {}

        print("Dataset building ...")
        super().__init__(data_dir=data_dir, device=device)

        self.vocab = list(self.word_vocab.vocab.keys())
        self.id2ner = self.ner_vocab.reverse_vocab
        self.id2word = self.word_vocab.reverse_vocab

        # split the data dataframe
        self.train_data_df = self.data_df.loc[self.data_df["split"] == "Train"]
        self.val_data_df = self.data_df.loc[self.data_df["split"] == "Val"]
        self.test_data_df = self.data_df.loc[self.data_df["split"] == "Test"]

        print("Building finished.\n")
        print("Get Sequences ...")
        # extract sequences from dataframes
        self.train = self.__get_sequence(self.train_data_df, self.ner_df)
        self.val = self.__get_sequence(self.val_data_df, self.ner_df)
        self.test = self.__get_sequence(self.test_data_df, self.ner_df)
        print("Finished.")

    def _parse_data(self, data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to
        # identify the separate functions needed.

        #  initialize lists for dataframe frames
        sentences_id = []
        tokens_id = []
        char_starts_id = []
        char_ends_id = []
        splits = []
        ners_id = []
        ners_sentences_id = []
        ners_char_starts_id = []
        ners_char_ends_id = []
        sentences_takeout_id = ["DDI-DrugBank.d238.s13",
                                "DDI-DrugBank.d238.s17",
                                "DDI-DrugBank.d238.s18",
                                "DDI-DrugBank.d238.s20",
                                "DDI-DrugBank.d238.s23",
                                "DDI-DrugBank.d216.s16",
                                "DDI-DrugBank.d270.s30",
                                "DDI-DrugBank.d270.s49",
                                "DDI-MedLine.d76.s9",
                                "DDI-DrugBank.d325.s7",
                                "DDI-DrugBank.d17.s6",
                                "DDI-MedLine.d137.s2",
                                "DDI-DrugBank.d54.s13",]
        
        # Get all xml files recursively
        allfiles = Path(data_dir).rglob("*.xml")

        # parse each xml file; get the required data:
        #   sentences_id:   <sentence> element's id found in xml
        #   split:          dataset split {TRAIN|VAL|TEST}
        #   tokens_id:      unique id that represents the token in our
        #                   vocabulary dictionary
        #   ner_id:         unique id that represents the NER label
        #   char_start_id:  token's start character position in the sentence
        #   char_end_id:    token's start character position in the sentence
        for myfile in allfiles:
            # parse the xml file
            tree = ET.parse(myfile)
            root = tree.getroot()
            # traverse the xml tree
            for sentence_e in root:

                sentence_id = sentence_e.get("id")
                if sentence_id in sentences_takeout_id:
                    continue
                
                # get sentence from xml
                # Add start and stop sequence for each sentence
                sentence_text = sentence_e.get("text").strip()
                sentence_text = start_sent_token + " " + sentence_text + " " + end_sent_token

                # tokenize sentence text
                mytokens, starts_id, ends_id = \
                    self.__tokenizer_w_ents(sentence_text)
                self.max_sample_length \
                    = max(self.max_sample_length, len(mytokens))

                # create vocabulary for tokens
                mytokens_id = [self.word_vocab.add(t) for t in mytokens]

                # save data info
                # get sentence id from xml
                # get the "split" field from the directory name, just under the data_dir
                # make two lists from "sentences_id" and "split" fields. The length = to the length of the tokens list
                sentences_id.extend([sentence_id]*len(mytokens))
                self.files_pathes[sentence_id] = myfile

                tokens_id.extend(mytokens_id)

                char_starts_id.extend(starts_id)
                char_ends_id.extend(ends_id)

                split = myfile.relative_to(data_dir).parts[0]
                if split == "Train":
                    split = random.choices(
                        ["Train", "Val"], weights=(71, 29))[0]
                splits.extend([split]*len(mytokens))

                # parse entities info
                ents_offset = []
                ents_label_ = []
                entities = []
                for element in sentence_e:
                    if element.tag == "entity":   # element could be of any type
                        # entity label
                        ents_label_.append(element.get("type"))
                        ents_offset.append(element.get("charOffset"))
                        entities.append(element.get("text"))

                # add sub label (prefix) to ner
                if entities:

                    # sub-labeling
                    ners_start, ners_end, ners_labels = \
                        self.__label_ner(ents_offset, ents_label_,
                                         starts_id, ends_id,
                                         )

                    # test ner sub-labeling, uncomment if needed
                    # self.__test_ner_labeling(   ents_label,
                    #                             ents_start, ents_ends, sentence_text, entities
                    #                         )

                    # create vocabulary for ner labels
                    ents_label_vocab = [self.ner_vocab.add(e)
                                        for e in ners_labels]

                    # save ners info

                    ners_sentences_id.extend(
                        [sentence_e.get("id")]*len(ents_label_vocab))
                    ners_id.extend(ents_label_vocab)
                    ners_char_starts_id.extend(ners_start.tolist())
                    ners_char_ends_id.extend(ners_end.tolist())

        #  build dataframes
        data_df_dict = {"sentence_id": sentences_id, "token_id": tokens_id,
                        "char_start_id": char_starts_id, "char_end_id": char_ends_id, "split": splits}
        ner_df_dict = {"sentence_id": ners_sentences_id, "ner_id": ners_id,
                       "char_start_id": ners_char_starts_id, "char_end_id": ners_char_ends_id}
        self.data_df = pd.DataFrame(data=data_df_dict)
        self.ner_df = pd.DataFrame(data=ner_df_dict)

    def __tokenizer_w_ents(self, text):
        # split as discussed in README.md
        # split on spaces and non-word characters; keeping the delimeters
        tokens_ = re.split(r'(\W)', text.lower())
        # get start and end position for each token
        st = 0
        end = 0
        starts = []
        ends = []
        for t in tokens_:
            end = st + len(t)
            if t.strip():   # avoid adding start & end positions of spaces
                starts.append(st)
                ends.append(end)
            # assert text[st:end].lower() == t
            st = end

        # remove spaces from tokens
        tokens = " ".join(tokens_).split()
        return tokens, starts, ends

    def __label_ner(self, ners_offset, ners_label, tokens_start, tokens_end):

        # add sublabel to ner labels. If ner has multiple token the first token label starts with I-, the next tokens starts with B-. One token ner's label always starts with I-

        # ners info: label, start and end chars positions.
        ners_lbl = []
        ners_st_id = []
        ners_end_id = []

        for offset, label in zip(ners_offset, ners_label):
            # get ner start/stop chars ids (from xml)
            pos_ = [p.split("-") for p in offset.split(";")]

            # if the ner has multiple non-consecutive tokens, then len(pos_)>1; i.e. i>0 (in the below loop)
            for i, pos in enumerate(pos_):
                # i>0; multiple non-consecutive tokens ner
                sublabels = ["I-", "I-"] if i > 0 else ["B-", "I-"]

                # We have start/stop indices for all tokens in the sentence. Now we want to get ner start/stop indices from this all tokens indices. (relate tokens with ner)
                # Find entity's start/stop chars ids (from xml) in tokens start/stop ids lists. i.e. search "pos" in  "tokens_start" and "tokens_end" lists
                # if the ner consists of multiple (consecutive) tokens, both of the start and the stop index will have the same location in "tokens_start" and "tokens_end" lists.
                # accounts for start sequence // add 9 to ids
                ner_st_id = tokens_start.index(int(pos[0])+9)
                # this piece of code handel the dataset issue discussed in README.md: section 1.3.2. Plural Entity
                if int(pos[1])+1+9 in tokens_end:
                    ner_end_id = tokens_end.index(int(pos[1])+1+9)
                elif int(pos[1])+1+1+9 in tokens_end:
                    ner_end_id = tokens_end.index(int(pos[1])+1+1+9)

                # multiple (non-consecutive or consecutive) tokens ner, sublabel depends on the value of i
                if ner_st_id != ner_end_id:
                    labels = [label]*(ner_end_id - ner_st_id + 1)
                    labels = [sublabels[0] + labels[0]] + \
                        [sublabels[1] + lb for lb in labels[1:]]

                # single token ner; non-consecutive ner
                elif i > 0:
                    labels = ["I-" + label]

                # single token ner
                else:
                    labels = ["B-" + label]

                # store and return ner info
                ners_lbl.extend(labels)
                ners_st_id.extend(tokens_start[ner_st_id:ner_end_id+1])
                ners_end_id.extend(tokens_end[ner_st_id:ner_end_id+1])

        ner_data = rfn.merge_arrays((ners_st_id, ners_end_id, ners_lbl))
        # token could have multiple tag, see README.MD.
        ner_data = np.unique(ner_data, axis=0)

        ner_start = ner_data[ner_data.dtype.names[0]]
        ner_end = ner_data[ner_data.dtype.names[1]]
        ner_labels = ner_data[ner_data.dtype.names[2]]
        #  check that there are no overllaping in start/stop ids
        # temp = np.unique(np.column_stack((ner_start, ner_end)).flatten())
        # assert np.all(temp[:-1] <= temp[1:])

        return ner_start, ner_end, ner_labels

    def __get_sequence(self, tokens_df, labels_df):
        # copy dfs
        labels_df_copy = labels_df.copy()
        tokens_df_copy = tokens_df.copy()
        # construct loc tuple = (char_start_id, char_start_id), then remove the char_start_id, char_start_id column from the df
        tokens_df_copy["loc"] = tokens_df_copy[["char_start_id",
                                                "char_end_id"]].apply(lambda x: tuple(x), axis=1)
        tokens_df_copy.drop(
            ["char_start_id",	"char_end_id"], inplace=True, axis=1)

        labels_df_copy["loc"] = labels_df_copy[["char_start_id",
                                                "char_end_id"]].apply(lambda x: tuple(x), axis=1)
        labels_df_copy.drop(
            ["char_start_id",	"char_end_id"], inplace=True, axis=1)

        O_label = self.ner_vocab.__getitem__("O")

        # merge two dfs based on the sentence_id and loc columns
        # A new column ner_id from the labels_df will have NaN if the sentence_id and loc are not found in the tokens_df
        # replace those NaNs with O_label
        sequence_df = tokens_df_copy.merge(
            labels_df_copy, on=["sentence_id", "loc"], how="left")
        sequence_df = sequence_df.assign(
            labels=lambda x: x.ner_id.fillna(O_label))
        
        # do some type casting
        sequence_df.astype({"token_id": np.int64, "labels": np.int64})
        sequence_df.drop(["split", "ner_id"], inplace=True, axis=1)

        # group dfs by sentence_id and aggregate to form a list for each column values
        sequence_df = sequence_df.groupby(["sentence_id"], as_index=False)[
            "token_id",	"loc",	"labels"].agg(lambda x: list(x))
        
        # add new column that hold a lengthe for each list
        sequence_df["length"] = sequence_df["token_id"].apply(lambda x: len(x))
        
        # change column names and convert to dict. For complience: needed as I change this code after i have finished
        sequence_df.columns = ["sent_id", "tokens", "loc", "labels", "length"]
        seq_data = sequence_df.to_dict(orient="list")
        
        return seq_data

    # def __group_arr(self, arr):
    #     data_group = {}
    #     _, ids = np.unique(arr[:, 0], return_index=True)
    #     ids = np.sort(ids)
    #     for i, j in zip(ids[0:-1], ids[1:]):
    #         data_group[arr[i, 0]] = arr[i:j, 1:]

    #     return data_group

    def load_iter(self):

        train_seq_df = pd.DataFrame(self.train)
        val_seq_df = pd.DataFrame(self.val)
        test_seq_df = pd.DataFrame(self.test)

        # create pytorch dataset
        train_pyt_ds = DatasetObject(train_seq_df)
        val_pyt_ds = DatasetObject(val_seq_df)
        test_pyt_ds = DatasetObject(test_seq_df)

        return train_pyt_ds, val_pyt_ds, test_pyt_ds

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        # get data

        # pad sequences to the max length
        max_len = self.max_sample_length
        pad_label = self.ner_vocab.__getitem__(pad_token)
        train_seq_label =  \
            pad_post_sequence(self.train["labels"],
                              max_len,
                              pad_label)
        val_seq_label =  \
            pad_post_sequence(self.val["labels"],
                              max_len,
                              pad_label)
        test_seq_label =  \
            pad_post_sequence(self.test["labels"],
                              max_len,
                              pad_label)
        # convert to tensors then save in device
        train_seq_label = torch.tensor(
            train_seq_label, dtype=torch.long, device=self.device)
        val_seq_label = torch.tensor(
            val_seq_label, dtype=torch.long, device=self.device)
        test_seq_label = torch.tensor(
            test_seq_label, dtype=torch.long, device=self.device)

        return train_seq_label, val_seq_label, test_seq_label

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        # get label sequence, count labels and store it in pandas.MultiIndex dataframe then plot
        matplotlib.style.use('ggplot')

        df_ner_train = self.ner_df.loc[self.ner_df["sentence_id"].isin(
            self.data_df["sentence_id"])]
        df_ner_val = self.ner_df.loc[self.ner_df["sentence_id"].isin(
            self.val_data_df["sentence_id"])]
        df_ner_test = self.ner_df.loc[self.ner_df["sentence_id"].isin(
            self.test_data_df["sentence_id"])]

        ner_dict = self.ner_vocab.vocab.copy()
        ner_dict.pop(pad_token)
        ner_dict.pop("O")

        data = {}
        for label, id_ in ner_dict.items():
            count = df_ner_train.loc[df_ner_train["ner_id"] == id_].shape[0]
            data[label] = count

        df = pd.DataFrame(data, index=["Train"])
        df = df.reindex(sorted(df.columns, key=lambda x: (
            x[2:], x[0]), reverse=True), axis=1)
        df.plot(kind="bar", figsize=(10, 10), rot=0, alpha=0.5)

        data = defaultdict(list)
        for split_df in [df_ner_val, df_ner_test]:
            for label, id_ in ner_dict.items():
                count = split_df.loc[split_df["ner_id"] == id_].shape[0]
                data[label].append(count)

        df = pd.DataFrame(data, index=["Val", "Test"])
        df = df.reindex(sorted(df.columns, key=lambda x: (
            x[2:], x[0]), reverse=True), axis=1)
        df.plot(kind="bar", figsize=(10, 10), rot=0)

    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        # extract sequences from dataframes

        matplotlib.style.use('ggplot')
        _, ax = plt.subplots(3, 1, figsize=(10, 10))

        df = pd.DataFrame(self.train_data_df["sentence_id"].value_counts())
        max_value = df.max().to_list()[0]
        bins = range(0, max_value, 5)
        xticks = range(0, max_value+1, 5)
        df.plot.hist(bins=bins, alpha=0.5, xticks=xticks, rot=90, ax=ax[0])
        ax[0].legend(["Train"])

        df = pd.DataFrame(self.val_data_df["sentence_id"].value_counts())
        max_value = df.max().to_list()[0]
        bins = range(0, max_value, 5)
        xticks = range(0, max_value+1, 5)
        df.plot.hist(bins=bins, alpha=0.5, xticks=xticks, rot=90, ax=ax[1])
        ax[1].legend(["Val"])

        df = pd.DataFrame(self.test_data_df["sentence_id"].value_counts())
        max_value = df.max().to_list()[0]
        bins = range(0, max_value, 5)
        xticks = range(0, max_value+1, 5)
        df.plot.hist(bins=bins, alpha=0.5, xticks=xticks, rot=90, ax=ax[2])
        ax[2].legend(["Test"])

    def plot_ner_per_sample_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        df_ner_train = self.ner_df.loc[self.ner_df["sentence_id"].isin(
            self.train_data_df["sentence_id"])]
        df_ner_val = self.ner_df.loc[self.ner_df["sentence_id"].isin(
            self.val_data_df["sentence_id"])]
        df_ner_test = self.ner_df.loc[self.ner_df["sentence_id"].isin(
            self.test_data_df["sentence_id"])]

        ner_train_count = df_ner_train.groupby(
            "sentence_id")["ner_id"].count().tolist()
        ner_val_count = df_ner_val.groupby(
            "sentence_id")["ner_id"].count().tolist()
        ner_test_count = df_ner_test.groupby(
            "sentence_id")["ner_id"].count().tolist()

        max_count = max(ner_train_count + ner_val_count + ner_test_count)

        df_train = pd.DataFrame({"Train": ner_train_count})
        df_val = pd.DataFrame({"Val": ner_val_count})
        df_test = pd.DataFrame({"Test": ner_test_count})

        matplotlib.style.use('ggplot')
        print("max count is: ", max_count)
        _, ax = plt.subplots(3, 1)
        bins = range(0, max_count, 3)
        xticks = range(0, max_count+4, 3)
        df_train.plot.hist(bins=bins, alpha=0.5, ax=ax[0], figsize=(
            10, 10), rot=90, xticks=xticks)

        df_val.plot.hist(bins=bins, alpha=0.5, ax=ax[1], figsize=(
            10, 10), rot=90, xticks=xticks)

        df_test.plot.hist(bins=bins, alpha=0.5, ax=ax[2], figsize=(
            10, 10), rot=90, xticks=xticks)

    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        ner_dict = self.ner_vocab.vocab.copy()
        ner_dict.pop(pad_token)
        ner_dict.pop("O")

        ner_group = {}
        for label, id_ in ner_dict.items():
            ner_group[id_] = label[2:]

        all_groups = defaultdict(set)
        for ner, grp in ner_group.items():
            df = self.ner_df[self.ner_df["ner_id"] == ner]
            sents = set(df["sentence_id"])
            all_groups[grp] = all_groups[grp].union(sents)

        venn(all_groups, fmt="{percentage:.1f}%", cmap="plasma", fontsize=10)

    def get_random_sample_1(self):
        split = random.choice(["train", "val", "test"])
        if split=="train":
            data = self.train
        elif split=="val":
            data = self.val
        else:
            data = self.test

        tokens_seq = data["tokens"]
        labels_seq = data["labels"]
        loc_seq = data["loc"]
        idx = random.choice(range(0, len(tokens_seq)-1))

        tokens = [self.id2word[x] for x in tokens_seq[idx]]
        labels = [self.id2ner[x] for x in labels_seq[idx]]
        loc = loc_seq[idx]
        sent = data["sent_id"][idx]

        # combine tokens to produce units that are seperated by spaces.
        units = self.__combine_tokens(tokens, loc)
        unit_label = self.__combine_labels(labels, loc)
        myfile = self.files_pathes[sent]

        print("--- Data parsed ---")
        display(pd.DataFrame(
            data={"tokens": units, "labels": unit_label}, 
            index=range(0, len(units)))
            )
        print("\n--- Data from xml file ---")
        print("file path: {}\n".format(str(myfile)))
        self.__get_data_xml(myfile, sent)

    def __combine_tokens(self, seq, loc):
        token_new = [seq[0]]
        start, end = zip(*loc)
        diff = [a - b for a, b in zip(end[:-1], start[1:])]
        for i, d in enumerate(diff):
            if d == 0:
                temp = token_new[-1] + seq[i+1]
                token_new[-1] = temp
            else:
                token_new.append(seq[i+1])

        return token_new

    def __combine_labels(self, seq, loc):
        token_new = [seq[0]]
        start, end = zip(*loc)
        diff = [a - b for a, b in zip(end[:-1], start[1:])]
        for i, d in enumerate(diff):
            if d == 0:
                temp = token_new[-1] + " | " + seq[i+1]
                token_new[-1] = temp
            else:
                token_new.append(seq[i+1])

        return token_new

    def __get_data_xml(self, myfile, sent_id):
        tree = ET.parse(myfile)
        root = tree.getroot()
        for sentence_e in root:
            if sentence_e.get("id") == sent_id:
                sentence_text = sentence_e.get("text")
                ents_label_ = []
                entities = []
                for element in sentence_e:
                    if element.tag == "entity":
                        ents_label_.append(element.get("type"))
                        entities.append(element.get("text"))

                print(sentence_text+"\n")

                display(pd.DataFrame(
                    data={"ner": entities, "label": ents_label_}, index=range(0, len(entities))))


# # to group the commented function
    # def __test_ner_labeling(self, ners_label, ners_start, ners_ends,
    #                         sentence_text, entities):

    #     #####  #####  #####  #####  #####  #####
    #     #   Does not work with the new logic   #
    #     #      see README.MD section           #
    #     #####  #####  #####  #####  #####  #####

    #     # Group ner by I-, to check that the labeling is ok
    #     ids = [[e] for e in ners_label if e.startswith("I-")]

    #     # extract ner from text using start and end char id
    #     ner_list = []
    #     for i,j in zip(ners_start, ners_ends):
    #         ner_list.append(sentence_text[i:j])

    #     # check that the extracted ner from sentence based on start and end ids are equal to the ner found in xml
    #     ner_extracted = "".join(ner_list)
    #     ner_xml = "".join(entities).replace(" ", "")
    #     if ner_xml != ner_extracted:
    #         diff = set()
    #         for di in ndiff(ner_extracted, ner_xml):
    #             if di[0] != ' ':
    #                 diff.add(di[2:])
    #         assert diff == {"s"}

    #     assert len(ids) == len(entities)

    # def __dicts_reform(self, labels_list):
    #     # get list of tuples (<sublabel, label>) and convert it to a dict to be used in two-levels pandas.MultiIndex dataframe
    #     key_level_0 = ["train", "val", "test"]
    #     pad_label = self.ner_vocab.__getitem__(pad_token)
    #     O_label = self.ner_vocab.__getitem__("O")

    #     dict_reform = {}
    #     for key, dict_ in zip(key_level_0, labels_list):
    #         dict_.pop(pad_label); dict_.pop(O_label)
    #         dict_decode_keys = [(self.ner_vocab.__getitem__(k), v)
    #             for k,v in dict_.items()]
    #         dict_decode_keys = sorted(dict_decode_keys, key=lambda x: x[0][2:])
    #         temp = {}
    #         for label, count in dict_decode_keys:
    #             main_label = label[2:]
    #             temp[(main_label, label)] = count

    #         dict_reform[key] = temp
    #     return dict_reform

def pad_post_sequence(seqs, max_len, pad_char):
    seq_pad = []
    for seq in seqs:
        len_diff = max_len - len(seq)
        padding = [pad_char] * len_diff

        seq_copy = seq[:]
        seq_copy.extend(padding)
        seq_pad.append(seq_copy)

    return seq_pad


def compare_df(df1, df2):
    arr1 = df1.values[:, 1:]
    arr2 = df2.values
    assert (arr1 == arr2).all()
