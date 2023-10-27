import numpy as np

from airsim.collections import RadarSystem, ControlPoint

rs1 = RadarSystem(position=np.array([10000, 10000, 0]), detection_radius=100000, error=10)
rs2 = RadarSystem(position=np.array([-10000, 10000, 0]), detection_radius=100000, error=10)
rs3 = RadarSystem(position=np.array([-10000, -10000, 0]), detection_radius=100000, error=10)
rs4 = RadarSystem(position=np.array([10000, -10000, 0]), detection_radius=100000, error=10)
rs5 = RadarSystem(position=np.array([10000, 10000, 0]), detection_radius=100000, error=10)
cp = ControlPoint()

print(rs1, rs2, rs3, rs4, rs5, cp)

cp.attach_radar_system(rs1)

print(rs1, rs2, rs3, rs4, rs5, cp)


cp.attach_radar_system(rs2)
cp.attach_radar_system(rs3)
cp.attach_radar_system(rs4)


print(rs1, rs2, rs3, rs4, rs5, cp)

cp.detach_radar_systems(rs2)
cp.detach_radar_systems(rs3)


print(rs1, rs2, rs3, rs4, rs5, cp)

cp.attach_radar_system(rs5)

print(rs1, rs2, rs3, rs4, rs5, cp)

cp.detach_radar_systems(rs5)
print(rs1, rs2, rs3, rs4, rs5, cp)
# print(rs1, rs2, rs3, rs4, rs5, cp)
