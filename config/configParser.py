import json

class ConfigParser(object):
    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)

        if config:
            self._update(config)

    def add(self, key, value):
        self.__dict__[key] = value
        
    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = ConfigParser(config[key])

            if isinstance(config[key], list):
                config[key] = [ConfigParser(x) if isinstance(x, dict) else x for x in config[key]]
            
        self.__dict__.update(config)

    def __repr__(self):
        return '%s' % self.__dict__


def main():
    config = ConfigParser(config_file='./sentimentConfig')
    print(config.corpus_info.train_data_file)


if '__main__' == __name__:
    main()

