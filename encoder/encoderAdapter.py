import sys
sys.path.append("../")
import tensorflow as tf
from config.configParser import ConfigParser
from encoder.rnnEncoder import RNNEncoder
from encoder.cnnEncoder import CNNEncoder
from encoder.dpCNNEncoder import DPCNNEncoder
from encoder.vsumEncoder import VSumEncoder, WeightedSumEncoder
from encoder.idCNNEncoder import IDCNNEncoder
from encoder.dCNNEncoder import DCNNEncoder
from encoder.attentionEncoder import AttentionEncoder
from encoder.hanEncoder import HANEncoder

"""
Encoder Adapter
"""

class EncoderAdapter(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.encoder_type = config.encoder_parameters.encoder_type if hasattr(config.encoder_parameters, "encoder_type") else "cnn_encoder"
        self.input_x = kwargs["input_x"] if "input_x" in kwargs else None
        self.encoder_options = {"rnn_encoder":RNNEncoder,
                                "cnn_encoder":CNNEncoder,
                                "dpcnn_encoder":DPCNNEncoder,
                                "vsum_encoder":VSumEncoder,
                                "weighted_sum_encoder":WeightedSumEncoder,
                                "idcnn_encoder":IDCNNEncoder,
                                "dcnn_encoder":DCNNEncoder,
                                "attention_encoder":AttentionEncoder,
                                "han_encoder":HANEncoder
                                }

    def getInstance(self):
        if self.encoder_type in self.encoder_options:
            print("EncoderType: " + self.encoder_type)
            return self.encoder_options[self.encoder_type](config=self.config, input_x=self.input_x)
        print("Encoder with default type: cnn_encoder")
        return self.encoder_options["cnn_encoder"](config=self.config, input_x=self.input_x)

if __name__ == "__main__":
    config = ConfigParser(config_file = "../config/sentimentConfig")
    encoder_adapter = EncoderAdapter(config)
    encoder = encoder_adapter.getInstance()
