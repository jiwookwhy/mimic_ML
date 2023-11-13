# To use this, just create an instance of Config in your file and access
# whichever members you need using the '.' operator
class Config():
    def __init__(self, **kwargs):
        # specify configurations specific to your setup here
        self.data_path = "parsing_scripts/raw_query_data"

        # specifies or overwrites existing configurations
        self.__dict__.update(kwargs)

