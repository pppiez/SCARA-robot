import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math

L1 = 10.0
L2 = 10.0
L3 = 5.0  # New link length for the third link

# Joint configuration
q = [np.pi / 4, np.pi / 3]

# Joint velocities
q_dot = [0.0, 0.5]

# Create a new figure and 3D axis object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(True)

# Initial joint angles for 3 links
q_sim = np.array([0.0, 0.0, 0.0])

# Initial joint velocities for 3 links
qd_sim = np.array([0.0, 0.0, 0.0])

# Set the joint angle of the third link to a constant value (e.g., 0.0)
q_sim[2] = 0.0

# Heart shape drawing parameters
num_frames = 1000
t = np.linspace(0, 2 * np.pi, num_frames)
heart_x = 8 * np.sin(t) ** 3
heart_y = 6.5 * np.cos(t) - 2.5 * np.cos(2 * t) - np.cos(3 * t) - 0.5 * np.cos(4 * t)

# Initialize plot elements
line, = ax.plot([], [], [], 'k-')
heart, = ax.plot([], [], [], 'rx')

J = np.array([
    [-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
    [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])]
])

# Calculate end-effector velocities
v = J @ q_dot
dt = 0.01
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
    """Compute the manipulability for a given Jacobian matrix."""
    w = np.sqrt(np.linalg.det(J @ J.T))
    return w

j = 1
end_effector_x = []
end_effector_y = []
end_effector_z = []

error_x = []
error_y = []

for j in range(len(t)):
    shift_x = 0  # กำหนด x-coordinate shift
    shift_y = 10  # กำหนด y-coordinate shift
    heart_x_shifted = heart_x[j] + shift_x # อัพเดตพิกัดหัวใจหลังจาก shift ตามแกน x
    heart_y_shifted = heart_y[j] + shift_y # อัพเดตพิกัดหัวใจหลังจาก shift ตามแกน y

    theta1, theta2 = inverse_kinematics_2joint(heart_x_shifted, heart_y_shifted, L1, L2)
    x_g = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y_g = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    j = j + 1
    print(j)

    while True:
        x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
        y = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
        # print(x,y)

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

        e = np.array([[e_x], [e_y]])
        print(abs(e_x) + abs(e_y))
        if (abs(e_x) + abs(e_y) <= 0.1):
            end_effector_x.append(x)
            end_effector_y.append(y)
            break

        # plot กราฟ
        ax.clear()
        ax.plot(end_effector_x, end_effector_y, 5, label='Path')
        ax.plot([base_x, base_x, L1 * np.cos(q[0]) + base_x, x + base_x, x + base_x], [base_y, base_y, L1 * np.sin(q[0]) + base_y, y + base_y, y + base_y], [0 ,10, 10, 10, 5], 'o-')  # Update plot (อิงจากตำแหน่ง base)
        ax.set_xlim([-L1 - L2, L1 + L2])
        ax.set_ylim([-L1 - L2, L1 + L2])
        ax.set_zlim([-L1 - L2, L1 + L2])
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.title("SCARA Robot simulation")
        m = compute_manipulability(J)
        plt.pause(0.0001)

plt.show()