#!/usr/bin/env python3
"""
ITU Rover odometri düğümü (düzeltilmiş sürüm)

Sorun: Araç skid-steer olduğu için sadece tekerlek hızlarından hesaplanan
yaw (omega = (vr - vl) / 0.89) gerçek dönüşten çok büyük çıkıyor ve yol
halkalar yapıyor.

Çözüm:
  - İleri hız (v)  -> tekerlek rpm'lerinden
  - Yönelim (yaw)  -> /imu1/data'dan (use_imu:=true, varsayılan)
  - IMU kullanılmazsa -> tekerlek omega'sı slip_factor ile düzeltilir
"""

import math
import threading

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf.transformations import quaternion_from_euler, euler_from_quaternion


# ROBOT PARAMETRELERİ

WHEEL_RADIUS = 0.135   # m
TRACK_WIDTH = 0.89     # m
UPDATE_RATE = 50       # Hz

# Sadece IMU kullanılmazsa geçerli: skid-steer kayma düzeltmesi (1.3 - 2.5)
DEFAULT_SLIP_FACTOR = 1.8

# Veri (thread güvenli)

lock = threading.Lock()
left_rpm = None
right_rpm = None
imu_yaw = None      # başlangıca göre göreli yaw (rad)
imu_yaw0 = None

# Robot durumu

x = 0.0
y = 0.0
yaw = 0.0
last_time = None

odom_pub = None
tf_broadcaster = None

use_imu = True
slip_factor = DEFAULT_SLIP_FACTOR


def normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rpm_to_velocity(rpm):
    return rpm * (2.0 * math.pi / 60.0) * WHEEL_RADIUS


# CALLBACKS

def left_callback(msg):
    global left_rpm
    if len(msg.data) < 2:
        rospy.logwarn_throttle(5, "Gecersiz sol teker verisi")
        return
    with lock:
        left_rpm = list(msg.data[:2])


def right_callback(msg):
    global right_rpm
    if len(msg.data) < 2:
        rospy.logwarn_throttle(5, "Gecersiz sag teker verisi")
        return
    with lock:
        right_rpm = list(msg.data[:2])


def imu_callback(msg):
    global imu_yaw, imu_yaw0
    q = msg.orientation
    _, _, raw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    with lock:
        if imu_yaw0 is None:
            imu_yaw0 = raw          # ilk okuma = yaw 0
        imu_yaw = normalize(raw - imu_yaw0)


# KİNEMATİK

def wheel_velocities(left, right):
    """Sol/sağ taraf lineer hızları (ön+arka ortalaması)."""
    v_left = (rpm_to_velocity(left[0]) + rpm_to_velocity(left[1])) / 2.0
    v_right = (rpm_to_velocity(right[0]) + rpm_to_velocity(right[1])) / 2.0
    return v_left, v_right


def integrate(v, new_yaw, dt):
    """Orta yön (mid-point) ile entegrasyon."""
    global x, y, yaw
    delta = normalize(new_yaw - yaw)
    mid = yaw + delta / 2.0
    x += v * dt * math.cos(mid)
    y += v * dt * math.sin(mid)
    yaw = normalize(yaw + delta)
    return delta / dt


# YAYIN

def publish(stamp, v, omega):
    q = quaternion_from_euler(0.0, 0.0, yaw)

    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"
    odom.pose.pose.position.x = x
    odom.pose.pose.position.y = y
    odom.pose.pose.orientation.x = q[0]
    odom.pose.pose.orientation.y = q[1]
    odom.pose.pose.orientation.z = q[2]
    odom.pose.pose.orientation.w = q[3]
    odom.twist.twist.linear.x = v
    odom.twist.twist.angular.z = omega
    odom_pub.publish(odom)

    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = "odom"
    t.child_frame_id = "base_link"
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    tf_broadcaster.sendTransform(t)


# ANA DÖNGÜ ADIMI

def step():
    global last_time

    with lock:
        if left_rpm is None or right_rpm is None:
            return
        if use_imu and imu_yaw is None:
            return
        left = left_rpm[:]
        right = right_rpm[:]
        current_imu_yaw = imu_yaw

    now = rospy.Time.now()
    if last_time is None:
        last_time = now
        return

    dt = (now - last_time).to_sec()
    if dt <= 0.0:
        return
    last_time = now

    v_left, v_right = wheel_velocities(left, right)
    v = (v_left + v_right) / 2.0

    if use_imu:
        new_yaw = current_imu_yaw
    else:
        omega_w = (v_right - v_left) / (TRACK_WIDTH * slip_factor)
        new_yaw = normalize(yaw + omega_w * dt)

    omega = integrate(v, new_yaw, dt)
    publish(now, v, omega)


def main():
    global odom_pub, tf_broadcaster, use_imu, slip_factor

    rospy.init_node("odometry_node")

    use_imu = rospy.get_param("~use_imu", True)
    slip_factor = rospy.get_param("~slip_factor", DEFAULT_SLIP_FACTOR)

    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
    tf_broadcaster = TransformBroadcaster()

    rospy.Subscriber("/drive_system_left_motors_feedbacks",
                     Float64MultiArray, left_callback)
    rospy.Subscriber("/drive_system_right_motors_feedbacks",
                     Float64MultiArray, right_callback)
    rospy.Subscriber("/imu1/data", Imu, imu_callback)

    rate = rospy.Rate(UPDATE_RATE)
    mode = "IMU yaw" if use_imu else "tekerlek yaw (slip=%.2f)" % slip_factor
    rospy.loginfo("Odometri basladi: %s", mode)

    while not rospy.is_shutdown():
        step()
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
