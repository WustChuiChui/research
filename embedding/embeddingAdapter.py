#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Embedding Adapter
"""

import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from config.configParser import ConfigParser
from common.baseLayer import BaseLayer
from embedding.wordEmbedding import WordEmbedding, RegionAlignmentLayer
from embedding.winPoolEmbedding import WinPoolEmbedding
from embedding.scalarRegionEmbedding import ScalarRegionEmbedding
from embedding.wordContextRegionEmbedding import WordContextRegionEmbedding
from embedding.contextWordRegionEmbedding import ContextWordRegionEmbedding
from embedding.multiRegionEmbedding import MultiRegionEmbedding

class EmbeddingAdapter(object):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.embedding_type = config.embedding_type
        self.embedding_options = {"word_embedding":WordEmbedding,
                                  "win_pool_embedding":WinPoolEmbedding,
                                  "scalar_region_embedding":ScalarRegionEmbedding,
                                  "word_context_embedding":WordContextRegionEmbedding,
                                  "context_word_embedding":ContextWordRegionEmbedding,
                                  "multi_region_embedding":MultiRegionEmbedding}
        self.region_size = config.region_size if hasattr(config, "region_size") else 3
        if self.embedding_type == "multi_region_embedding":
            region_size_list = config.region_size.strip().split(",")
            self.region_size = [int(item.strip()) for item in region_size_list]
        print(config)

    def getInstance(self):
        if self.embedding_type == "word_embedding":
            print("EmbeddingType: " + self.embedding_type)
            return self.embedding_options["word_embedding"](self.vocab_size, self.embedding_size)
        elif self.embedding_type in self.embedding_options:
            print("EmbeddingType: " + self.embedding_type)
            return self.embedding_options[self.embedding_type](self.vocab_size, self.embedding_size, self.region_size)
        else:
            print("EmbeddingType with default: " + self.embedding_type)
            return self.embedding_options["word_embedding"](self.vocab_size, self.embedding_size)

if __name__ == "__main__":
    config = ConfigParser(config_file = "../config/sentimentConfig")
    print(config)
    embedding_adapter = EmbeddingAdapter(config.model_parameters)
    embedding = embedding_adapter.getInstance()

