import bpy
import os
import math
from mathutils import Vector

# !!! PASTE THIS CODE INTO BLENDER (Scripting tab) !!!
# =========================
# 1. USER SETTINGS
# =========================
file_name="run-20260413_133124-k8t9e6kq_lambda_sem_adv_2_semantic"
original_file="reals/real_last_scale.obj"
generated_file="objects/last/0.obj"
obj_path = f"/Users/zaiyrsharsheyev/Documents/TU/World-R3GAN/output_test/wandb/{file_name}/files/arbitrary_random_samples_v1.00000_h1.00000_st0/{original_file}"

# If obj_path is relative, Blender will resolve it from the current .blend location
# obj_path = bpy.path.abspath(obj_path)

# Output video path
output_video = f"/Users/zaiyrsharsheyev/Documents/TU/World-R3GAN/output_test/wandb/{file_name}/files/orbit.mp4"

# Animation settings
frame_start = 1
frame_end = 240
fps = 24

# Camera settings
height_factor = 0.6
distance_factor = 2.2

# =========================
# 2. CLEAN SCENE
# =========================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# =========================
# 3. IMPORT OBJ
# =========================
if not os.path.exists(obj_path):
    raise FileNotFoundError(f"OBJ file not found: {obj_path}")

before = set(bpy.context.scene.objects)

try:
    bpy.ops.wm.obj_import(filepath=obj_path)
except Exception:
    bpy.ops.import_scene.obj(filepath=obj_path)

after = set(bpy.context.scene.objects)
imported_objects = list(after - before)

if not imported_objects:
    raise RuntimeError("No objects were imported from the OBJ file")

mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
if not mesh_objects:
    raise RuntimeError("Imported file contains no mesh objects")

# =========================
# 4. COMPUTE MODEL BOUNDS
# =========================
min_corner = Vector((float('inf'), float('inf'), float('inf')))
max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))

for obj in mesh_objects:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        min_corner.x = min(min_corner.x, world_corner.x)
        min_corner.y = min(min_corner.y, world_corner.y)
        min_corner.z = min(min_corner.z, world_corner.z)
        max_corner.x = max(max_corner.x, world_corner.x)
        max_corner.y = max(max_corner.y, world_corner.y)
        max_corner.z = max(max_corner.z, world_corner.z)

center = (min_corner + max_corner) / 2
size = max_corner - min_corner
max_dim = max(size.x, size.y, size.z)

if max_dim <= 0:
    max_dim = 1.0

# =========================
# 5. CREATE ORBIT CENTER
# =========================
bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
orbit_center = bpy.context.object
orbit_center.name = "OrbitCenter"

# =========================
# 6. ADD CAMERA
# =========================
cam_distance = max_dim * distance_factor
cam_height = center.z + max_dim * height_factor

cam_location = (center.x, center.y - cam_distance, cam_height)
bpy.ops.object.camera_add(location=cam_location)
camera = bpy.context.object
camera.name = "OrbitCamera"

# Parent camera to empty so rotating the empty rotates camera around model
camera.parent = orbit_center

# Make camera always look at the center
track = camera.constraints.new(type='TRACK_TO')
track.target = orbit_center
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# Set as active camera
bpy.context.scene.camera = camera

# =========================
# 7. ADD LIGHTS
# =========================
bpy.ops.object.light_add(type='SUN', location=(center.x + max_dim, center.y - max_dim, center.z + max_dim * 2))
sun = bpy.context.object
sun.data.energy = 3.0

bpy.ops.object.light_add(type='AREA', location=(center.x, center.y + max_dim, center.z + max_dim))
area = bpy.context.object
area.data.energy = 1000
area.data.shape = 'RECTANGLE'
area.scale = (max_dim, max_dim, 1)

# =========================
# 8. ANIMATE ORBIT
# =========================
scene = bpy.context.scene
scene.frame_start = frame_start
scene.frame_end = frame_end
scene.render.fps = fps

orbit_center.rotation_euler = (0, 0, 0)
orbit_center.keyframe_insert(data_path="rotation_euler", frame=frame_start)

# Slightly less than 360 for smoother loop
orbit_center.rotation_euler = (0, 0, math.radians(359.0))
orbit_center.keyframe_insert(data_path="rotation_euler", frame=frame_end)

# Linear interpolation
if orbit_center.animation_data and orbit_center.animation_data.action:
    for fcurve in orbit_center.animation_data.action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'LINEAR'

# =========================
# 9. RENDER SETTINGS
# =========================
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
scene.render.filepath = output_video

print("OBJ loaded from:", obj_path)
print("Video will be saved to:", output_video)

# Белый фон
world = bpy.context.scene.world
world.use_nodes = True

bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)  # белый цвет

# Uncomment this line if you want the script to start rendering immediately
bpy.ops.render.render(animation=True)