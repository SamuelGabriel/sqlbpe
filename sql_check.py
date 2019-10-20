import sqlite3

from utils import flatten_bpe_encodings
from sqlbpe.tokenize_sql import untokenise


class QuerySyntaxChecker():
    def __init__(self, vocab):
        self.vocab = vocab
        self.temp_db = sqlite3.connect(":memory:")
    
    def string_is_syntactical(self, query):
        try:
            self.temp_db.execute(query)
        except sqlite3.OperationalError as e:
            e_str = str(e)
            # This check accepts all statements in all datasets looked at
            # it does not provide full guidance though for statements with ALL syntax
            if 'no such table' not in e_str and 'near \"ALL\"' not in e_str:
                return False
        return True
    
    def tokenised_is_syntactical(self, tok_query):
        query = untokenise(' '.join(t if type(t) is str else 'intdummy' for t in tok_query)) # str(t) 
        return self.string_is_syntactical(query)

    def is_syntactical(self, prediction):
        tok_query = flatten_bpe_encodings(prediction, self.vocab)
        return self.tokenised_is_syntactical(tok_query)
        