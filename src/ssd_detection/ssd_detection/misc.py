# ros2_ws/src/ssd_detection/ssd_detection/misc.py
import time

class Timer:
    def __init__(self):
        self.clock = dict()
		
		# Start inference
    def start(self, key="default"):
        self.clock[key] = time.time()
		
		# End inference
    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        fps = time.time() - self.clock[key] # (Start - End)
        del self.clock[key]
        
        return fps
