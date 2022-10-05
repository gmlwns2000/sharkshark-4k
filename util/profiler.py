import time

class Profiler:
    def __init__(self) -> None:
        self.start_ticks = {}
        self.data = {}
    
    def set(self, name, value):
        self.data[name] = value
    
    def start(self, name):
        self.start_ticks[name] = time.time()

    def end(self, name):
        if name in self.start_ticks:
            ticks = self.start_ticks[name]
            del self.start_ticks[name]
            elapsed = time.time() - ticks
            self.data[name] = elapsed
        else:
            elapsed = -1
            #print(f'Profiler: "{name}" is not started region.')
        
        return elapsed