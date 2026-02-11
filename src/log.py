class Logger:
    def __init__(self, *files) -> None:
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush() 
    
    def flush(self):
        for f in self.files:
            f.flush()
