# Sim2Real (alpha)
In order to integrate the real robot SDK you need to:
1. Edit the `model/robots/robot.py` script adding the call(s) to your SDK. Check line `308`, there is a TODO comment with more info. 
2. Check the values in `model/robots/ROBOT_NAME/motor_constants.py`. They should match your real robot motors values (e.g. kp/kd, directions)
3. Run the `playground.py` script:
```
robot-gym playground
```
4. Use the GUI to change the controlleryou want to use. You can also plug in a gamepad.

For more parameters and flags run:
```
robot-gym playground --help
```

I would suggest adding a security stop function. Check `io/gamepad/xbox_one_pad.py` line 78.

PLEASE NOTE THIS SOFTWARE WASN'T TESTED ON ANY REAL ROBOT. USE AT YOUR OWN RISK!