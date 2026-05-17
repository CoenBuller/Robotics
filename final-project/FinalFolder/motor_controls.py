import picar_4wd as fc 
from typing import Optional


class MotorControl():
    def __init__(self, power_forward: int=20, power_backward: int=20, power_steer: int=5, classes: Optional[dict[int, str]] = None, high_notes: Optional[set[str]] = None, motor_controls: bool = True):
        
        self.power_forward = power_forward
        self.power_backward = power_backward
        self.power_steer = power_steer
        self.motor_controls = motor_controls

        self._forward: bool = False
        self._backward: bool = False
        self._turning: bool = False

        if classes is None:
            self.moves = {0 : "none", 1: "turn", 2: "forward", 3: "backward"}
        else: 
            self.moves = classes

        if high_notes is None:
            self.high_notes = set(['D#', 'E', 'F', 'F#', 'G', 'G#'])
        else: 
            self.high_notes = high_notes
    
    @property 
    def turning(self):
        return self._turning
    
    @turning.setter
    def turning(self, value):
        assert isinstance(value, bool), "Turning value should be a boolean"
        self._turning = value

    @property
    def backward(self):
        return self._backward

    @backward.setter
    def backward(self, value):
        assert isinstance(value, bool), "Backward value should be a boolean"
        self._backward = value


    @property 
    def forward(self):
        return self._forward
    
    @forward.setter
    def forward(self, value):
        assert isinstance(value, bool), "Forward value should ba boolean"
        self._forward = value

    def _detect_move(self, label, note):
        move = self.moves[label]
        if move == "none":
            return None
        
        if move == "turn":
            if note in self.high_notes:
                return "right"
            else:
                return "left"
            
        return move 


    def _pick_action(self, move):

        if move in ("left", "right"):
            return move

        if self.backward and move == "forward":
            self.backward = False
            return "stop"

        if self.forward and move == "backward":
            self.forward = False
            return "stop"

        if not self.forward and move == "forward":
            self.forward = True
            self.backward = False
            return move

        if not self.backward and move == "backward":
            self.backward = True
            self.forward = False
            return move

        return None
    def _execute_action(self, action):

        if action == "forward":
            self.turning = False
            fc.forward(self.power_forward)

        elif action == "backward":
            self.turning = False
            fc.backward(self.power_backward)

        elif action == "left":
            self.turning = True
            fc.turn_left(self.power_steer)

        elif action == "right":
            self.turning = True
            fc.turn_right(self.power_steer)

        elif action == "stop":
            self.turning = False
            fc.stop()


    def __call__(self, label, note):
        move = self._detect_move(label=label, note=note)
        if move is None:
            return 

        action = self._pick_action(move=move)
        self._execute_action(action=action)
