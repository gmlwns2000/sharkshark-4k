import time

class Profiler:
    def __init__(self) -> None:
        self.start_ticks = {}
        self.data = {}
        self.elapsed_ticks = {}
    
    def set(self, name, value):
        self.data[name] = value
    
    def start(self, name):
        self.start_ticks[name] = time.time()

    def end(self, name):
        if name in self.start_ticks:
            ticks = self.start_ticks[name]
            del self.start_ticks[name]
            elapsed = time.time() - ticks
            previous_elapsed = self.elapsed_ticks.get(name, (0,0))
            self.elapsed_ticks[name] = (previous_elapsed[0] + elapsed, previous_elapsed[1] + 1)
            self.data[name] = (self.elapsed_ticks[name][0] / self.elapsed_ticks[name][1])
        else:
            elapsed = -1
            #print(f'Profiler: "{name}" is not started region.')
        
        return elapsed