# -*- coding: utf-8 -*-

from index.es import ElasticsearchManager


class UtteranceManager(ElasticsearchManager):

    def __init__(self, host="localhost:9200"):
        ElasticsearchManager.__init__(self, host)
        self.index = "dbdc3-dev-en"
        self.doc_type = "utternace"
        self.properties = {
            "corpus": {
                "type": "string",
                "store": "true",
                "index": "not_analyzed",
                "term_vector": "with_positions_offsets",
            },
            "utterances": {
                "type": "nested",
            },
            "text": {
                "type": "string",
                "store": "true",
                "index": "analyzed",
                "term_vector": "with_positions_offsets",
            },
        }


class DialogueManager(ElasticsearchManager):

    def __init__(self, host="localhost:9200"):
        ElasticsearchManager.__init__(self, host)
        self.index = "dbdc3-dev-en"
        self.doc_type = "dialogue"
        self.properties = {
            "corpus": {
                "type": "string",
                "store": "true",
                "index": "not_analyzed",
                "term_vector": "with_positions_offsets",
            },
            "dialogue-id": {
                "type": "",
            },
            "utterances": {
                "type": "nested",
            },
            "text": {
                "type": "string",
                "store": "true",
                "index": "analyzed",
                "term_vector": "with_positions_offsets",
            },
        }
