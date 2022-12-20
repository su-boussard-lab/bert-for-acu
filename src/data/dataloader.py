"""
File used to process loading the existing data.
"""
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import torch
from src.utils.logger import log
from torch.utils.data import TensorDataset, DataLoader
from src.data.dataset import CustomTextDataset, MultiModalDataset
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import IPython
import torch.nn.functional as F
import re


class ACUDataLoader:
    def __init__(
        self,
        data_path: str,
        label_path: str,
        label_type: str,
        test_ids: str,
        train_ids: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        max_words: int = 2000,
        test_only: bool = False,
        use_tabular: bool = False,
        tab_path: str = None,
        ordinal_regression: bool = False,
    ):
        """Initializes the dataloader class
        Args:
            data_path (str): path to the language data
            label_path (str): path to the labels
            label_type (str): type of labels for ACU prediction
            test_ids (str): path to the test ids
            train_ids (str): path to the training ids
            tokenizer (AutoTokenizer): tokenizer applied for BERT
            max_length (int, 512): maximum sequence length of a chunk
            max_words (int, 2000): maximum words considered, from the start for memory efficiency
            test_only (bool, False): loads only the test data
            use_tabular (bool, False): loads the tabular data for a multimodal model
            tab_path (str, None): path to the tabular data
            ordinal_regression (bool, False): make ordinal regression model, rather then simple regression
        Returns:
            DataLoader instance
        """
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.label_path = label_path
        self.label_type = label_type
        self.test_ids = test_ids
        self.train_ids = train_ids
        self.max_length = max_length
        self.max_words = max_words
        self.test_only = test_only
        self.use_tabular = use_tabular
        self.tab_path = tab_path
        self.ordinal_regression = ordinal_regression

    def get_train_test_data(self) -> Tuple:
        """Retrieve the train and validation data
        Args:
        Returns:
            train (torch.Dataset): training dataset
            test (torch.Dataset): testing dataset
            len (int): total length
        """
        train, train_labels = get_data(
            self.data_path,
            self.label_path,
            self.label_type,
            self.train_ids,
            self.use_tabular,
            self.tab_path,
            self.ordinal_regression,
        )
        test, test_labels = get_data(
            self.data_path,
            self.label_path,
            self.label_type,
            self.test_ids,
            self.use_tabular,
            self.tab_path,
            self.ordinal_regression,
        )

        label_list = label_list = list(
            set(train_labels.values.tolist() + test_labels.values.tolist())
        )

        if self.use_tabular:
            train, test, self.tabular_data_size = scale_data(train, test)

        if self.test_only:
            train = None
        else:
            train = get_dataset(
                train,
                train_labels,
                label_list,
                self.tokenizer,
                self.max_length,
                self.max_words,
                self.use_tabular,
            )

        test = get_dataset(
            test,
            test_labels,
            label_list,
            self.tokenizer,
            self.max_length,
            self.max_words,
            self.use_tabular,
        )

        return train, test, len(label_list)

    def convert_ids_to_vector(
        self, data, model, batch_size: int = 1, device: str = "cpu"
    ) -> TensorDataset:
        """Converts ids to vectors for the model
        Args:
            data:
            model:
            batch_size (int, 1):
            device (str, "cpu"):
        Returns:
            TensorDataset
        """
        model.eval()

        dataLoader = DataLoader(data, batch_size=batch_size, shuffle=False)
        progress_bar = tqdm(dataLoader, desc="boosting: ", unit="dataset")
        pooled_vector = []
        with torch.no_grad():
            for idx, batch in enumerate(progress_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_id = batch
                pooled_output = model.ids_to_vector(
                    input_ids=input_ids, segment_ids=segment_ids, input_mask=input_mask
                )
                pooled_vector.append(pooled_output.to("cpu"))
        pooled_vector = torch.cat(pooled_vector)
        dummy_data = torch.zeros(len(data))
        progress_bar.close()

        model.train()
        return TensorDataset(pooled_vector, dummy_data, dummy_data, data.tensors[3])

    def get_train_val_data(
        self, train_complete: TensorDataset, val_size: float = 0.2
    ) -> Tuple:
        """Creates a validation set from the existing train data
        Args:
            train_complete (TensorDataset): training dataset
            val_size (float, 0.2): percentage of the training dataset taken for validation
        Returns:
            train (TensorDataset): new reduced training dataset
            val (TensorDataset): validation dataset
        """
        num_train = len(train_complete)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(val_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train = torch.utils.data.Subset(train_complete, train_idx)
        valid = torch.utils.data.Subset(train_complete, valid_idx)
        return train, valid

    def get_tabular_size(self) -> int:
        """Returns the size of the tabular data, if called
        Args:
            None
        Returns:
            feature length of tabular data
        """
        return self.tabular_data_size


def get_data(
    data_path: str,
    labels_path: str,
    label_type: str,
    ids_path: str,
    use_tabular: bool,
    tab_path: str,
    ordinal_regression: bool,
) -> Tuple:
    """Gets the data from a path
    Args:
        data_path (str): path to the data
        labels_path (str): path to the labels dataframe
        label_type (str): label type
        ids_path (str): path to the [training, test] ids
        use_tabular (bool): if true, then tabular data is addded to the dataset for multimodality
        tab_path (str): path to tabular data
        ordinal_regression (bool): apply ordinal regression on labels
    Returns:
        data (list): list of the features
        label_list (list): list of labels
    """
    data = pd.read_csv(data_path, engine="python").set_index("PAT_DEID")
    labels = pd.read_csv(labels_path, engine="python").set_index("PAT_DEID")

    if ordinal_regression:
        label_type = [c for c in labels.columns if "ANY" in c and not "EVER" in c]

    labels = labels[label_type].reindex(data.index)

    if ordinal_regression:
        labels = labels.sum(axis=1)

    # Load only the given IDs
    ids = pd.read_csv(ids_path, engine="python").PAT_DEID
    data = data.loc[data.index.intersection(ids)]
    labels = labels.loc[data.index.intersection(ids)]

    if use_tabular:
        tab_data = (
            pd.read_csv(tab_path, engine="python")
            .set_index("PAT_DEID")
            .drop("DEMO_INDEX_PRE_CHE", axis=1)
        )
        data = data.join(tab_data, how="left")

    # return data, labels
    return data.iloc[:10], labels.iloc[:10]


def get_dataset(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    labels_list: List,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    max_words: int = 2000,
    use_tab: bool = False,
) -> TensorDataset:
    """Create a Tensordataset from the list of labels and data
    Args:
        data (pd.DataFrame): dataframe of features
        labels (pd.DataFrame): dataframe of labels
        labels_list (List): list set of labels
        tokenizer (AutoTokenizer): tokenization tool
        max_length (int, 512): maximum length of the tokenization list
        max_words (int, 2000): maximum words considered, for memory space reasons.
        use_tab (bool, False): if True, we use a multimodal dataset with tabular data
    Returns:
        Tensordataset of the features and labels"""
    features = convert_examples_to_features(
        data_df=data.note,
        labels_df=labels,
        label_list=labels_list,
        tokenizer=tokenizer,
        max_length=max_length,
        max_words=max_words,
    )

    all_input_dicts = [f.input_dict for f in features]
    all_input_dicts = {
        k: torch.cat([dic[k] for dic in all_input_dicts]) for k in all_input_dicts[0]
    }
    all_label_ids = torch.Tensor([f.label_id for f in features])
    all_chunks_per_notes = torch.Tensor([f.chunks_per_note for f in features])

    if use_tab:
        return MultiModalDataset(
            all_input_dicts,
            data.drop("note", axis=1).values,
            all_label_ids,
            all_chunks_per_notes,
        )
    else:
        return CustomTextDataset(
            all_input_dicts,
            all_label_ids,
            all_chunks_per_notes,
        )


def convert_examples_to_features(
    data_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_list: List,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    max_words: int = 2000,
) -> List:
    """Convert examples into features
    Args:
        data(pd.DataFrame): dataframe of free text
        label (pd.DataFrame): dataframe of labels
        label_list (List): list set of labels
        tokenizer (AutoTokenizer): tokenization tool
        max_length (int, 512): maximum length of the input vectors
        max_words (int, 2000): maximum words considered, for memory space reasons.
    Returns:
        features (List): list of input features
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for idx, (data, label) in enumerate(zip(data_df, labels_df)):
        if idx % 1000 == 0:
            print("Writing example %d" % (idx))

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #

        # truncate data length to maximum number of sentences:
        """
        encoded_input = tokenizer(
            data.split("\n"),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        """
        encoded_input = tokenizer(
            preprocess_text(data.lower()),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )

        step_chunks = encoded_input.pop("overflow_to_sample_mapping")

        encoded_input = {k: v[:max_words] for k, v in encoded_input.items()}
        chunks_per_note = encoded_input["input_ids"].shape[0]

        # Pad all the sequences to a constant length

        label_id = label_map[label]

        features.append(
            InputFeatures(
                input_dict=encoded_input,
                label_id=label_id,
                chunks_per_note=chunks_per_note,
            )
        )
    return features


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """Scale the data to be zero-center and unit-variance.
        NOTE: the scaler is only fitted on the training data, and then all the data is transformed,
        This is done to prevent dataleakage in the test set
    Args:
        X_train (pd.DataFrame): training tabular data that is not yet scaled.
        X_test (pd.DataFrame): testing tabular data that is not yet scaled.
    Returns:
        scaled_data (pd.DataFrame): data frame with normalized data for test and training set
        tabular_data_size (int): size of the features of the tabular data
    """
    scaler = StandardScaler()
    col_names = [c for c in X_train.columns if c != "note"]
    tabular_data_size = len(col_names)
    scaler.fit(X_train[col_names])
    X_train_ = scaler.transform(X_train[col_names])
    X_train[col_names] = X_train_
    X_test_ = scaler.transform(X_test[col_names])
    X_test[col_names] = X_test_

    X_train = pd.DataFrame(data=X_train, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(data=X_test, columns=X_test.columns, index=X_test.index)

    return X_train, X_test, tabular_data_size


def _truncate_seq(tokens_a: List, max_length: int) -> None:
    """Helper function to trunkate a sequence in place
    Args:
        tokens_a (List): the initial list
        max_lenth (int): point at which the list is truncated
    Returns:
        None
    """
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()


def preprocess_text(x: str) -> str:
    """Preprocesses the BERT text very simply
    Args:
        x (str): input string, unprocessed
    Returns:
        y (str): processed output string
    """
    y = re.sub("\\[(.*?)\\]", "", x)  # remove de-identified brackets
    y = re.sub(
        "[0-9]+\.", "", y
    )  # remove 1.2. since the segmenter segments based on this
    y = re.sub("dr\.", "doctor", y)
    y = re.sub("m\.d\.", "md", y)
    y = re.sub("admission date:", "", y)
    y = re.sub("discharge date:", "", y)
    y = re.sub("--|__|==", "", y)
    return y


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_dict, label_id, chunks_per_note):
        self.input_dict = input_dict
        self.label_id = label_id
        self.chunks_per_note = chunks_per_note
