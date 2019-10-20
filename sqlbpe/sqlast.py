import sqlparse
from .tokenize_sql import tokenise, untokenise

class SqlAst():
    def __init__(self, query):
        self.ast = self._post_process(self._parse_subquery(sqlparse.parse(self._pre_process(query))[0]))
    
    def get_ast(self):
        return self.ast
    
    def get_flattened(self, join_char=' '):
        return self._flatten(self.ast, join_char)

    def get_all_possible_neighbor_pairs(self, join_char=' '):
        return set(self._get_all_neighbor_pairs_of_descendants(self.ast, join_char))
    
    @classmethod
    def _get_all_neighbor_pairs_of_descendants(cls, subtree, join_char=' '):
        if isinstance(subtree, str):
            return []
        else:
            return sum([cls._get_all_neighbors_of_childs(subtree, join_char)] + [cls._get_all_neighbor_pairs_of_descendants(c, join_char) for c in subtree], [])

    @classmethod    
    def _get_all_neighbors_of_childs(cls, subtree, join_char=' '):
        if isinstance(subtree, str):
            return []
        num_childs = len(subtree)
        res = []
        for pair_len in range(2,num_childs+1):
            for offset in range(0, num_childs-pair_len+1):
                for split_idx in range(1, pair_len):
                    res.append((cls._flatten(subtree[offset:offset+split_idx], join_char), cls._flatten(subtree[offset+split_idx:offset+pair_len], join_char)))
        return res

    @classmethod
    def _flatten(cls, ast, join_char=' '):
        return join_char.join(cls._flatten(c, join_char) for c in ast) if not isinstance(ast, str) else ast

    @classmethod
    def _parse_subquery(cls, sub_query):
        if hasattr(sub_query, 'tokens'):
            if len(sub_query.tokens) == 1:
                return cls._parse_subquery(sub_query.tokens[0])
            else:
                return [cls._parse_subquery(t) for t in sub_query.tokens if t.ttype != sqlparse.tokens.Token.Text.Whitespace]
        else:
            return sub_query.value
    
    @classmethod
    def _pre_process(cls, query):
        return untokenise(query)

    @classmethod
    def _post_process(cls, ast):
        if isinstance(ast,str):
            tokenised_ast = [x for x in tokenise(ast).split(' ') if x]
            return tokenised_ast[0] if len(tokenised_ast) == 1 else tokenised_ast
        else:
            return [cls._post_process(c) for c in ast]

def get_restrictions(listofseqs, join_char='__'):
    sql_asts = [SqlAst(target) for target in listofseqs]
    restrict_bpe_pairs_to = [sa.get_all_possible_neighbor_pairs(join_char=join_char) for sa in sql_asts]
    return restrict_bpe_pairs_to
