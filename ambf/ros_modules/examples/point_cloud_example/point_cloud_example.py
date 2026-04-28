#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2025, AMBF
#     (https://github.com/WPI-AIM/ambf)
#
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:
#
#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#
#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.
#
#     \author    <amunawa2@jh.edu>
#     \author    Adnan Munawar
#     \version   1.0
# */
# //==============================================================================

from ros_abstraction_layer import ral
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32
import time
import numpy as np
import math


print('INFO! MAKE SURE TO RUN THE ADF FILE IN THIS FOLDER CONTAINING THE POINT CLOUD OBJECTS!')

ral = ral('test_pc')

time.sleep(1.0)

pub1 = ral.publisher('/ambf/env/point_clouds/point_cloud_one', PointCloud, queue_size=10)

pub2 = ral.publisher('/ambf/env/point_clouds/point_cloud_two', PointCloud, queue_size=10)

msg = PointCloud()
msg.header.frame_id = '/ambf/env/BODY Chassis'

num_points = 10000
for i in range(num_points):
    msg.points.append(Point32())

slp = 0.01
size_msg = Float32()
size_msg.data = 4.0

while not ral.is_shutdown():
    cnt_i = int(np.sqrt(num_points))
    cnt_j = int(np.sqrt(num_points))
    delta_th = 2. * np.pi / cnt_i
    r_offset = (0.2 * math.sin(ral.to_sec(ral.now())))
    r = 0.4 + r_offset
    for i in range(cnt_i):
        th = i * delta_th
        for j in range(cnt_j):
            phi = j * delta_th
            idx = i * cnt_j + j
            msg.points[idx].x = r * math.cos(th) * math.sin(phi)
            msg.points[idx].y = r * math.sin(th) * math.sin(phi)
            msg.points[idx].z = r * math.cos(phi)

    pub1.publish(msg)
    time.sleep(slp)

    cnt_i = int(np.sqrt(num_points))
    cnt_j = int(np.sqrt(num_points))
    delta_th = 2. * np.pi / cnt_i
    r = 0.1 + r_offset / 5.0
    R = 0.5 + r_offset
    for i in range(cnt_i):
        th = i * delta_th
        for j in range(cnt_j):
            phi = j * delta_th
            idx = i * cnt_i + j
            msg.points[idx].x = (R + r * math.cos(th)) * math.cos(phi)
            msg.points[idx].y = (R + r * math.cos(th)) * math.sin(phi)
            msg.points[idx].z = r * math.sin(th)

            # This part can be commented out. It is just for cool rotating donut effect
            t = ral.to_sec(ral.now())
            ct = math.cos(t)
            st = math.sin(t)
            x = msg.points[idx].x
            z = msg.points[idx].z
            # Rotation around the Y axis
            msg.points[idx].x = x * ct + z * st
            msg.points[idx].z = -x * st + z * ct
            x = msg.points[idx].x
            y = msg.points[idx].y
            # Further Rotation around the Z axis
            msg.points[idx].x = x * ct - y * st
            msg.points[idx].y = x * st + y * ct

    pub2.publish(msg)
    time.sleep(slp)
