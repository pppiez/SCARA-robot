import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from mpl_toolkits.mplot3d import Axes3D

L1 = 8.0
L2 = 8.0

# Joint configuration
q = [np.pi/4, np.pi/3]

# Joint velocities
q_dot = [0.0, 0.5]

# สร้าง figure และ axis
fig, ax = plt.subplots()
ax.grid(True)

# กำหนดองศาของ joint เริ่มต้น
q_sim = np.array([0.0, 0.0])

# กำหนด joint velocities เริ่มต้น
qd_sim = np.array([0.0, 0.0])

# Heart shape drawing parameters
num_frames = 1000 # กำหนด step ที่ใช้ในการวาดรูปหัวใจ
t = np.linspace(0, 2 * np.pi, num_frames)
# gain ด้วย 0.5 เพื่อสเกลขนาดหัวใจลง
heart_x = 0.5*(8 * np.sin(t) ** 3) # สมการการวาดหัวใจแกน x
heart_y = 0.5*(6.5 * np.cos(t) - 2.5 * np.cos(2*t) - np.cos(3*t) - 0.5 * np.cos(4*t)) # สมการการวาดหัวใจแกน y
# heart_x = 8 * np.sin(t+5) ** 3
# heart_y = 6.5 * np.cos(t+5) - 2.5 * np.cos(2*(t+5)) - np.cos(3*(t+5)) - 0.5 * np.cos(4*(t+5))


# สร้าง Jacobian matrix
J = np.array([
    [-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
    [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])]
])

# หา end-effector velocities
v = J @ q_dot
dt = 0.01

# PID
k_p = 10
k_i = 0.2
k_d = 0
integral_x = 0
integral_y = 0
prev_error_x = 0
prev_error_y = 0

# Base
base_x = 0
base_y = 0

# คำนวณ inverse kinematic สำหรับ 2 joint
def inverse_kinematics_2joint(x, y, l1, l2):
    # คำนวณระยะทางจากจุด origin(base ของหุ่นยนต์) ไปยัง target point
    distance = np.sqrt(x**2 + y**2)

    # ตรวจสอบว่า target point สามารถไปถึงได้ด้วย robot นี้หรือไม่
    if distance > l1 + l2 or distance < np.abs(l1 - l2):
        print("Target point is out of reach")
        return None

    # คำนวณมุมจาก แกน x(ทาง +) ไปยัง target point
    theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
    
    # คำนวณองศาระหว่าง line connecting(origin), target point และ line perpendicular ไปยัง L1
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    return theta1, theta2

def compute_manipulability(J):
    # คำนวณ manipulability สำหรับ Jacobian matrix
    w = np.sqrt(np.linalg.det(J @ J.T))
    if(w < 0):
        print('Singularity')
    return w

# Function to calculate forward kinematics for a 2-joint planar robot
def forward_kinematics_2joint(theta1, theta2, link1_length, link2_length):
    x = link1_length * np.cos(theta1) + link2_length * np.cos(theta1 + theta2)
    y = link1_length * np.sin(theta1) + link2_length * np.sin(theta1 + theta2)
    return x, y


j = 1
end_effector_x = []
end_effector_y = []
theta1_value = []
theta2_value = []

# ทำซ้ำตามจำนวน timestep
for j in range(len(t)):
    # Shift start point
    shift_x = 0  # กำหนด x-coordinate shift
    shift_y = 10  # กำหนด y-coordinate shift
    heart_x_shifted = heart_x[j] + shift_x # อัพเดตพิกัดหัวใจหลังจาก shift ตามแกน x
    heart_y_shifted = heart_y[j] + shift_y # อัพเดตพิกัดหัวใจหลังจาก shift ตามแกน y

    theta1, theta2 = inverse_kinematics_2joint(heart_x_shifted, heart_y_shifted, L1, L2) # หา inverse kinematic

    # หาตำแหน่ง Goal ในแนวแกน x, y
    x_g, y_g = forward_kinematics_2joint(theta1, theta2, L1, L2)
    j = j + 1 # อัพเดตค่า j
    # print(j)

    while True: # วนซ้ำจนกว่า ค่าพิกัดที่ถูกจูนผ่าน PID จะมีค่า error ของค่า goal เทียบค่าปัจจุบัน น้อยกว่าค่าที่กำหนด
        # หาตำแหน่ง x, y ปัจจุบัน
        x, y = forward_kinematics_2joint(q[0], q[1], L1, L2)
        # print(x,y)

        # Jacobian matrix
        J = np.array([
            [-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
            [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])]
        ])
        J_1 = J[:, 0]
        J_2 = J[:, 1]

        # Proportional term
        e_x = x_g - x
        e_y = y_g - y

        # Integral term
        integral_x += e_x * dt
        integral_y += e_y * dt

        # Derivative term
        derivative_x = (e_x - prev_error_x) / dt
        derivative_y = (e_y - prev_error_y) / dt

        # คำนวณร่วมกับ PID terms (Control signal)
        u_x = k_p * e_x + k_i * integral_x + k_d * derivative_x
        u_y = k_p * e_y + k_i * integral_y + k_d * derivative_y

        # อัพเดต previous error สำหรับ derivative term
        prev_error_x = e_x
        prev_error_y = e_y

        # อัพเดต positions จาก control signal
        J_pseudo_inv = np.linalg.pinv(J)
        q_dot = J_pseudo_inv @ np.array([[u_x], [u_y]])
        q[0] = (q[0] + (q_dot[0] * dt))[0]
        q[1] = (q[1] + (q_dot[1] * dt))[0]

        # พิจารณา error
        e = np.array([[e_x], [e_y]])
        # print(abs(e_x) + abs(e_y))
        # ถ้าผลรวมของ error ทั้งแกน x และ y มีค่าน้อยกว่า 0.15 ให้ break ออกไปพิจารณาพิกัดการวาดหัวใใจถัดไป
        if(abs(e_x) + abs(e_y) <= 0.15):
            end_effector_x.append(x)
            end_effector_y.append(y)

            theta1_value.append(np.cos(q[0]))
            theta2_value.append(np.sin(q[0]))

            print(theta1)
            print(theta2)
            break

# Function to visualize the movement of the robot arm
def plot_robot_arm(theta1_values, theta2_values, link1_length, link2_length):
    # plt.figure(figsize=(8, 6))
    plt.title('SCARA Robot movement')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    workspace_x = []
    workspace_y = []

    for i in range(len(theta1_values)):
        x, y = forward_kinematics_2joint(theta1_values[i], theta2_values[i], link1_length, link2_length)
        workspace_x.append(x)
        workspace_y.append(y)

    plt.scatter(workspace_x, workspace_y, c='blue', s=2, label='Workspace')        
    plt.plot(0, 0, 'ko')  # Plotting the base of the robot arm
    plt.plot(end_effector_x,end_effector_y,label = 'Path')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

# Example usage:
def run_robot_arm():
    # Lengths of the robot links
    link1_length = 8
    link2_length = 8
    
    # Calculate workspace by generating theta values
    num_steps = 400
    theta1_values = np.linspace(-180, 180, num_steps)  # Range of theta1 values
    theta2_values = np.linspace(-180, 180, num_steps)  # Range of theta2 values

    plot_robot_arm(theta1_values, theta2_values, link1_length, link2_length)

# Run the code
run_robot_arm()