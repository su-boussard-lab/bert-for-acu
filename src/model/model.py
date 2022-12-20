"""
Base model file, which defines every model
"""


from transformers import BertModel
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import numpy as np
import torch.nn.functional as F
from src.layers.position_embedding import SinusoidalPositionalEmbedding
from src.layers.ordinal_regression import LogisticCumulativeLink
from transformers import GPT2Model, BertModel, DistilBertModel, LongformerModel


class ACUClassifier(nn.Module):
    """Create an ACU NLP prediction classifier
    Args:
        bert (BertModel): bert backbone architecture
        num_class (int): number of classes
        bert_finetuning (bool): BERT layers are frozen or not
        dropout (float): dropout probability
        cls_pooling (bool): true if only to pool on the CLS token, otherwise pools on full embedding
    """

    def __init__(
        self,
        bert: BertModel,
        num_class: int,
        bert_finetuning: bool,
        dropout_p: float,
        cls_pooling: bool,
        intermediate_mlp_size: int = 256,
        ordinal_regression: bool = False,
        **kwargs,
    ) -> None:
        super(ACUClassifier, self).__init__()
        self.bert = bert
        self.dropout_p = dropout_p
        self.num_class = num_class
        self.bert_finetuning = bert_finetuning
        self.cls_pooling = cls_pooling
        self.intermediate_mlp_size = intermediate_mlp_size
        self.skip_MLP = True if self.intermediate_mlp_size == -1 else False
        self.ordinal_regression = ordinal_regression

        bert_hidden_size = self.bert.config.hidden_size

        # intermediate small MLP
        if self.skip_MLP:
            self.intermediate_mlp_size = bert_hidden_size
        else:
            self.MLP = nn.Sequential(
                nn.Linear(bert_hidden_size, self.intermediate_mlp_size * 2),
                nn.Dropout(p=dropout_p),
                # nn.BatchNorm1d(self.intermediate_mlp_size * 2),
                nn.ReLU(),
                nn.Linear(self.intermediate_mlp_size * 2, self.intermediate_mlp_size),
                nn.Dropout(p=dropout_p),
                # nn.BatchNorm1d(self.intermediate_mlp_size),
                nn.ReLU(),
            )

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.intermediate_mlp_size, 1)

        if self.ordinal_regression:
            self.ordinallink = LogisticCumulativeLink(self.num_class)
        else:
            self.sigmoid = nn.Sigmoid()

        if not self.bert_finetuning:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

    def ids_to_vector(
        self,
        input_dicts: torch.Tensor,
        boost: bool = False,
        return_sequence: bool = False,
    ) -> torch.FloatTensor:
        """
        Args:
            input_dicts (torch.Tensor): contains the input_ids and potentially also the attention and segmentations masks
            boost (bool, False): for debugging
            return_sequence (bool, False): returns the BERT ouput without going through the pooler
        Returns:
            `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
                classifier pretrained on top of the hidden state associated to the first character of the
                input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
                [batch_size, hidden_size]
        """
        if isinstance(self.bert, DistilBertModel):
            input_dicts.pop("token_type_ids")

        output = self.bert(**input_dicts)
        if return_sequence:
            if isinstance(self.bert, BertModel):
                return output[0]
            elif isinstance(self.bert, DistilBertModel):
                return output[0]
            elif isinstance(self.bert, GPT2Model):
                return output.last_hidden_state
            else:
                raise NotImplementedError

        else:
            if isinstance(self.bert, BertModel):
                return output[1]
            elif isinstance(self.bert, DistilBertModel):
                return output[0][:, 0, :]
            elif isinstance(self.bert, GPT2Model):
                return output.last_hidden_state[:, 0, :]
            elif isinstance(self.bert, LongformerModel):
                return output.pooler_output
            else:
                raise NotImplementedError

    def preprocess(self, list_input_dicts: dict) -> tuple:
        """Preprocesses the input dictionaries, sucht that they have the adequate shape to put into the transformer

            B: Batch size
            network_inputs: 2,3 depending on network -> inputs_ids, attention_masks, segmentation_ids
            chunks: number of chunks
            max_length: predefined in the beginning. Can also be shorter if n_chunks = 1
        Args:
            list_input_dicts: list of input dicts
        returns:
            new_dicts (dict): new shape: [sum chunks in the whole batch, max_length]
            step_chunnks (torch.Tensory): a np array of length sum_of_chunks with the according indices
        """
        new_dicts = {}
        input_keys = list_input_dicts[0].keys()
        for key in input_keys:
            stacked_inputs = []
            step_chunks = []
            for i, input_dicts in enumerate(list_input_dicts):
                stacked_inputs.append(input_dicts[key])
                step_chunks.extend([i] * len(input_dicts[key]))
            # Dim: [sum n_chunks, max_length]
            new_dicts[key] = torch.cat(stacked_inputs, dim=0).to(self.bert.device)
        return new_dicts, torch.Tensor(np.array(step_chunks)).to(
            self.bert.device, dtype=torch.int64
        )

    def postprocess(
        self,
        embeddings: torch.Tensor,
        attention_masks: torch.Tensor,
        step_chunks: torch.Tensor,
        cls_pooling: bool = True,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Postprocesses the output embeddings, such that they correspond to their input dimension
           Average the chunks either on the CLS token or pool them generally
            Input Dim: [sum chunks, emb_dim]
            Output Dim: [batchsize, emb_dim]
        Args:
            embeddings (torch.Tensor): tensor of the embeddings
            attention_masks (torch.Tensor): tensor of the attention masks
            step_chunks (torch.Tensor): tensor containing the chunk indices per note
            cls_pool (bool, True): if averaging the chunks only on the CLS token or over the whole note
            return_sequence (bool, False): returns the full sequence rather than only the CLS or averaged token
        returns:
            new_dicts (dict): new shape: [sum chunks in the whole batch, max_length]
        """
        if return_sequence:
            return scatter_mean(embeddings, step_chunks, dim=0)

        # Pool only the CLS token at the beggining of each sentence
        if cls_pooling:
            embeddings = embeddings[:, 0, :]

        # Pool all the sequences together and take their average weighted by the mask
        else:
            input_mask_expanded = (
                attention_masks.unsqueeze(-1).expand(embeddings.size()).float()
            )
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        embeddings_pooled = scatter_mean(embeddings, step_chunks, dim=0)
        return embeddings_pooled

    def get_L1_loss(self) -> torch.Tensor:
        """Computes the L1 norm of the parameters in the fully connected layer
        Args:
            None
        Returns:
            torch.Tensor with a scalar of the L1 norm of the FC layer
        """
        params = torch.cat([x.view(-1) for x in self.fc.parameters()])
        L1 = torch.norm(params, p=1)
        return L1

    def transformer_forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunks: torch.Tensor,
        boost: bool = False,
        return_sequence: bool = False,
        run_MLP: bool = True,
        indipendent_chunks: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the BERT backbone and simple MLP
        Args:
            input_ids (torch.Tensor): input token ids
            token_type_ids (torch.Tensor): type of tokens
            attention_mask (torch.Tensor): attention mask
            chunks (torch.Tensor): chunks per note
            boost (bool, False): for debugging
            return_sequence (bool, False): returns the full sequence rather than only the CLS or averaged token
            run_MLP (bool, True): runs the embeddings trough a smaller MLP
            indipendent_chunks (bool, False): if the chunks are seen as independent and not averaged
        Returns
            MLP output
        """
        # input_dicts, step_chunks = self.preprocess(list_input_dicts=input_dicts)
        batch_size = len(chunks)
        input_dicts = {
            "input_ids": input_ids.to(self.bert.device),
            "token_type_ids": token_type_ids.to(self.bert.device),
            "attention_mask": attention_mask.to(self.bert.device),
        }

        # Pass through the BERT network
        x = self.ids_to_vector(input_dicts, boost, return_sequence=return_sequence)

        # Do a scatter mean
        if not indipendent_chunks:
            step_chunks = []
            for i in range(batch_size):
                step_chunks += [i] * int(chunks[i])
            step_chunks = torch.Tensor(step_chunks).to(
                self.bert.device, dtype=torch.int64
            )

            x = self.postprocess(
                x,
                input_dicts["attention_mask"],
                step_chunks,
                self.cls_pooling,
                return_sequence=True,
            )

        if run_MLP:
            x = self.MLP(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunks: torch.Tensor,
        indipendent_chunks: bool = False,
        boost: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the BERT backbone and the final classifier
        Args:
            input_ids (torch.Tensor): input token ids
            token_type_ids (torch.Tensor): type of tokens
            attention_mask (torch.Tensor): attention mask
            chunks (torch.Tensor): chunks per note
            indipendent_chunks (bool, False): if the chunks are seen as independent and not averaged
            boost (bool, False): for debugging
        Returns
            logit (torch.Tensor): logit output
        """
        # input_ids : (B, D)
        # B : batch size,
        # D : dimenstion of tokens
        x = self.transformer_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            chunks=chunks,
            run_MLP=not self.skip_MLP,
            indipendent_chunks=indipendent_chunks,
        )
        logit = self.fc(x)
        if self.ordinal_regression:
            return self.ordinallink(logit)
        risk = self.sigmoid(logit)
        return risk[:, 0]


class ACUMultiModalClassifier(ACUClassifier):
    """Create an ACU NLP prediction classifier
    Args:
        bert (BertModel): bert backbone architecture
        num_class (int): number of classes
        bert_finetuning (bool): BERT layers are frozen or not
        dropout (float): dropout probability
        cls_pooling (bool): true if only to pool on the CLS token, otherwise pools on full embedding
    """

    def __init__(
        self,
        bert: BertModel,
        num_class: int,
        bert_finetuning: bool,
        dropout_p: float,
        cls_pooling: bool,
        tabular_size: int,
        intermediate_mlp_size: int = 256,
        ordinal_regression: bool = False,
    ) -> None:
        print("##################### Load 'Multimodal Logistic ACU Classifier' Model")
        super().__init__(
            bert,
            num_class,
            bert_finetuning,
            dropout_p,
            cls_pooling,
            intermediate_mlp_size,
            ordinal_regression,
        )
        self.fc = nn.Linear(self.intermediate_mlp_size + tabular_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_x: torch.Tensor,
        chunks: torch.Tensor,
        boost: bool = False,
        indipendent_chunks: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the BERT backbone and the final classifier
        Args:
            input_ids (torch.Tensor): input token ids
            token_type_ids (torch.Tensor): type of tokens
            attention_mask (torch.Tensor): attention mask
            tabular_x (torch.Tensor): tabular data
            chunks (torch.Tensor): chunks per note
            boost (bool, False): for debugging
            indipendent_chunks (bool, False): if the chunks are seen as independent and not averaged
        Returns
            logit (torch.Tensor): logit output
        """
        # process the NLP features first through BERT and the MLP
        x = self.transformer_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            chunks=chunks,
            run_MLP=not self.skip_MLP,
            indipendent_chunks=indipendent_chunks,
        )

        # D : dimenstion of bert hidden - 256 + 760 tabular data
        if indipendent_chunks:
            tabular_x_new = []
            for i, tab in enumerate(tabular_x):
                tabular_x_new += [tab] * int(chunks[i])
            tabular_x = torch.stack(tabular_x_new).to(self.bert.device)
        x = torch.cat([x, tabular_x.to(self.bert.device)], axis=1)

        x = self.dropout(x)
        logit = self.fc(x)
        if self.ordinal_regression:
            return self.ordinallink(logit)
        risk = self.sigmoid(logit)
        return risk[:, 0]


class ACUMultiModalAttentionClassifier(ACUClassifier):
    """Create an ACU NLP prediction classifier that uses cross modality attention
    NOTE: this model does not take into account the whole NLP sequence, only the CLS output after and MLP
    Args:
        bert (BertModel): bert backbone architecture
        num_class (int): number of classes
        bert_finetuning (bool): BERT layers are frozen or not
        dropout (float): dropout probability
        cls_pooling (bool): true if only to pool on the CLS token, otherwise pools on full embedding
        tabular_size (int): dimension size of the tabular data
    """

    def __init__(
        self,
        bert: BertModel,
        num_class: int,
        bert_finetuning: bool,
        dropout_p: float,
        cls_pooling: bool,
        tabular_size: int,
        intermediate_mlp_size: int = 256,
        ordinal_regression: bool = False,
    ) -> None:
        print(
            "##################### Load 'Multimodal Cross-Attention ACU Classifier' Model"
        )
        super().__init__(
            bert,
            num_class,
            bert_finetuning,
            dropout_p,
            cls_pooling,
            intermediate_mlp_size,
            ordinal_regression,
        )
        self.tabular_size = tabular_size

        # define cross-modality transformers
        self.trans_nlp_with_tab = self.get_network(
            self_type="nlp",
            tab_size=self.tabular_size,
            nlp_size=self.intermediate_mlp_size,
        )
        self.trans_tab_with_nlp = self.get_network(
            self_type="tab",
            tab_size=self.tabular_size,
            nlp_size=self.intermediate_mlp_size,
        )
        self.fc = nn.Linear(self.intermediate_mlp_size + self.tabular_size, 1)

    def get_network(self, self_type: str, tab_size: int, nlp_size: int) -> nn.Module:
        """Creates a Multihead Attention Network that does cross-modality attention
        Args:
            self_type (str): either "tab" for tabular network, or "nlp" for language network
            tab_size (int): dimensions of the tabular input
            nlp_size (int): dimensions of the language input
        Returns:
            torch.nn.Module of the multihead attention
        """
        if self_type == "nlp":
            embed_dim, kdim, vdim = (nlp_size, tab_size, tab_size)
            num_heads = 4
        elif self_type == "tab":
            embed_dim, kdim, vdim = (tab_size, nlp_size, nlp_size)
            num_heads = 1
        else:
            raise ValueError(f"'{self_type}' is an unknown network type")

        return torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=self.dropout_p,
            batch_first=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_x: torch.Tensor,
        chunks: torch.Tensor,
        boost: bool = False,
        indipendent_chunks: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the BERT backbone and the final classifier
        Args:
            input_ids (torch.Tensor): input token ids
            token_type_ids (torch.Tensor): type of tokens
            attention_mask (torch.Tensor): attention mask
            tabular_x (torch.Tensor): tabular data
            chunks (torch.Tensor): chunks per note
            boost (bool, False): for debugging
            indipendent_chunks (bool, False): if the chunks are seen as independent and not averaged
        Returns
            logit (torch.Tensor): logit output
        """
        tabular_x = tabular_x.unsqueeze(1).to(self.bert.device)

        # process the NLP features first through BERT and the MLP
        nlp_x = self.transformer_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            chunks=chunks,
            return_sequence=False,
            run_MLP=not self.skip_MLP,
            indipendent_chunks=indipendent_chunks,
        ).unsqueeze(1)

        # Do cross modality attention
        if indipendent_chunks:
            tabular_x_new = []
            for i, tab in enumerate(tabular_x):
                tabular_x_new += [tab] * int(chunks[i])
            tabular_x = torch.stack(tabular_x_new).to(self.bert.device)

        tab_to_nlp, _ = self.trans_nlp_with_tab(nlp_x, tabular_x, tabular_x)
        nlp_to_tab, _ = self.trans_tab_with_nlp(tabular_x, nlp_x, nlp_x)

        # concatenate
        x = torch.cat([tab_to_nlp[:, 0, :], nlp_to_tab[:, 0, :]], axis=1)

        # final logistic regression layer
        x = self.dropout(x)
        logit = self.fc(x)
        if self.ordinal_regression:
            return self.ordinallink(logit)
        risk = self.sigmoid(logit)
        return risk[:, 0]


class ACUMMSequenceAttentionClassifier(ACUMultiModalAttentionClassifier):
    """Create an ACU NLP prediction classifier that uses cross modality attention taking into account the whole sequence
    NOTE: this model takes into account the whole NLP sequence, computational cost is increased
    Args:
        bert (BertModel): bert backbone architecture
        num_class (int): number of classes
        bert_finetuning (bool): BERT layers are frozen or not
        dropout (float): dropout probability
        cls_pooling (bool): true if only to pool on the CLS token, otherwise pools on full embedding
        tabular_size (int): dimension size of the tabular data
    """

    def __init__(
        self,
        bert: BertModel,
        num_class: int,
        bert_finetuning: bool,
        dropout_p: float,
        cls_pooling: bool,
        tabular_size: int,
        ordinal_regression: bool = False,
        **kwargs,
    ) -> None:
        print(
            "##################### Load 'Multimodal Cross-Attention ACU Classifier with Sequence' Model"
        )
        super().__init__(
            bert,
            num_class,
            bert_finetuning,
            dropout_p,
            cls_pooling,
            tabular_size,
            ordinal_regression,
        )
        self.bert_hidden_size = self.bert.config.hidden_size
        self.tabular_size = tabular_size

        # define cross-modality transformers
        self.embed_positions = SinusoidalPositionalEmbedding(self.bert_hidden_size)
        self.trans_nlp_with_tab = self.get_network(
            self_type="nlp",
            tab_size=self.tabular_size,
            nlp_size=self.bert_hidden_size,
        )
        self.trans_tab_with_nlp = self.get_network(
            self_type="tab",
            tab_size=self.tabular_size,
            nlp_size=self.bert_hidden_size,
        )

        del self.MLP
        self.fc = nn.Linear(self.bert_hidden_size + tabular_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_x: torch.Tensor,
        chunks: torch.Tensor,
        boost: bool = False,
        indipendent_chunks: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the BERT backbone and the final classifier
        Args:
            input_ids (torch.Tensor): input token ids
            token_type_ids (torch.Tensor): type of tokens
            attention_mask (torch.Tensor): attention mask
            tabular_x (torch.Tensor): tabular data
            chunks (torch.Tensor): chunks per note
            boost (bool, False): for debugging
            indipendent_chunks (bool, False): if the chunks are seen as independent and not averaged
        Returns
            logit (torch.Tensor): logit output
        """
        if indipendent_chunks:
            tabular_x_new = []
            for i, tab in enumerate(tabular_x):
                tabular_x_new += [tab] * int(chunks[i])
            tabular_x = torch.stack(tabular_x_new).to(self.bert.device)

        tabular_x = tabular_x.unsqueeze(1).to(self.bert.device)

        # process the NLP features first through BERT and the MLP
        nlp_x = self.transformer_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            chunks=chunks,
            return_sequence=True,
            run_MLP=False,
            indipendent_chunks=indipendent_chunks,
        )

        # Add positional embedding
        nlp_x += self.embed_positions(nlp_x[:, :, 0])

        # Do cross modality attention
        tab_to_nlp, _ = self.trans_nlp_with_tab(nlp_x, tabular_x, tabular_x)
        nlp_to_tab, _ = self.trans_tab_with_nlp(tabular_x, nlp_x, nlp_x)

        # concatenate
        x = torch.cat([tab_to_nlp[:, 0, :], nlp_to_tab[:, 0, :]], axis=1)

        # final logistic regression layer
        x = self.dropout(x)
        logit = self.fc(x)
        if self.ordinal_regression:
            return self.ordinallink(logit)
        risk = self.sigmoid(logit)
        return risk[:, 0]
