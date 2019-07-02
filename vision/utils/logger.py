class Logger:
    def __init__(self, tag):
        self.tag = tag
    
    # verbose
    def v(self, msg):
        print(f'{self.tag}: {msg}')