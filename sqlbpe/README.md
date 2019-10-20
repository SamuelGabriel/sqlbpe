## SQL BPE

In this direcotyr you can find the main code to use BPE on SQL.

The code works on the tokenized datasets from [text2sql-data](https://github.com/jkkummerfeld/text2sql-data). AST SQL first detokenizes the data, and then builds an AST with the help of sqlparse.

The main file `bpe.py` works similar to torchtext Vocabulary in building a vocabulary of type `bpe.BPEVocab` which has a similar API to `torchtext.vocab.Vocab`. It assumes to be provided an object with both a `itos` and a `stoi` attribute, like `Vocab`.

To run BPE without restriction to the SQL AST:

Initialize `bpe.Bpe` object with your token-level vocabulary and call fit with list of lists containing your token ids. You can also hand-over a list of tokens and specify `vocab_map=True`.
> bpe_handler = bpe.Bpe(vocab)
> bpe_handler.fit(listofseqs, t=100)

To run BPE with the stopping criterion use
> bpe_handler.fit_to_valid(listofseqs, val_listofseqs, retention_count=<r from the paper, good choice: 20>, min_freq=<m from the paper, good choice: 100>)

To use AST BPE you first need to get the restrictions on the kinds of things that can represent nodes in the BPE tree
> restrictions = {'train': sqlast.get_restrictions(listofseqs), 'val': sqlast.get_restrictions(val_listofseqs)}

and then specifying these restrictions when calling either `fit` or `fit_to_valid`.

