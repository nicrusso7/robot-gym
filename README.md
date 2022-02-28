# robot-gym
RL and classical techniques applied to robotics.



#requirements python3.7

#Installation

Create a Python 3.7 virtual environment, e.g. using Anaconda

``conda create -n [envname] python=3.7 anaconda``

``conda activate [envname]``


#Install from source

Clone this repository and run from the root of the project:

``git clone https://github.com/nicrusso7/robot-gym.git``

``cd robot-gym``

``git checkout k3lso-stable``

``pip install .``





Notes for shapely

``conda config --add channels conda-forge ``

``conda install shapely``



#Notes

``In robot-gym/robot_gym/model/robots/simple_motor.py``
The ``(motor_commands)`` on line 133 outputs a 60 dimension vector (q, qdot, kp, kd, tau) = (Postion, velocities, PID gains, motor torque) for each motor. 






