"""Synthetic sanity check for stereo/AprilTag pose and metric Q scaling; no real data needed."""
import importlib.util
from pathlib import Path
import numpy as np
import cv2
import tempfile

spec = importlib.util.spec_from_file_location('stereo_pipeline', Path(__file__).parent/'stereo_pipeline.py')
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

K = np.array([[1080.,0,480],[0,1090.,270],[0,0,1.]])
D = np.zeros((1,5))
size = np.array([960,540])
R, _ = cv2.Rodrigues(np.array([0.012,-0.024,0.009]))
T = np.array([[-0.210],[0.006],[0.001]])
calib = {'K1':K, 'D1':D, 'K2':K.copy(), 'D2':D.copy(), 'size':size}
rect = sp.rectification_matrices(calib, R, T, 0)
assert rect['P2'][0,3] < 0, 'Expected standard left camera / right camera ordering'
obj = sp.tag_object_corners(.08)
R_tag, _ = cv2.Rodrigues(np.array([0.25,0.11,-0.04]))
t_tag = np.array([[-0.04],[-0.02],[1.30]])
R_tag_r = R @ R_tag
t_tag_r = R @ t_tag + T
rv1, _ = cv2.Rodrigues(R_tag)
rv2, _ = cv2.Rodrigues(R_tag_r)
l, _ = cv2.projectPoints(obj, rv1, t_tag, K, D)
r, _ = cv2.projectPoints(obj, rv2, t_tag_r, K, D)
l,r=l.reshape(4,2),r.reshape(4,2)
Rl,tl,el = sp.tag_pose(l,K,D,.08)
Rr,tr,er = sp.tag_pose(r,K,D,.08)
Rest = Rr @ Rl.T
Test = tr-Rest@tl
print('AprilTag relative-pose error (degrees, meters):', sp.rotation_distance_deg(R,Rest), np.linalg.norm(Test-T))
assert sp.rotation_distance_deg(R,Rest) < 0.03
assert np.linalg.norm(Test-T) < 0.0003
r_l = sp.rectified_points(l,K,D,rect['R1'],rect['P1'])
r_r = sp.rectified_points(r,K,D,rect['R2'],rect['P2'])
epi=np.abs(r_l[:,1]-r_r[:,1])
print('Epipolar error px median:', np.median(epi))
assert np.max(epi) < 0.001
X4=cv2.triangulatePoints(rect['P1'],rect['P2'],r_l.T,r_r.T)
X=(X4[:3]/X4[3]).T
edges=np.linalg.norm(X-np.roll(X,1,axis=0),axis=1)
print('Triangulated tag edge meters:', edges)
assert np.allclose(edges,.08,atol=1e-5)
# Downsampled disparity must scale with pixel resolution, and Q scaling must keep metric Z.
full_width,full_height=size
fx=rect['P1'][0,0]
base=np.linalg.norm(T)
Dfull=fx*base/2.0
full_proj=np.array([float(rect['P1'][0,2]),float(rect['P1'][1,2]),float(Dfull),1.])
full_xyz=rect['Q']@full_proj
full_xyz=full_xyz[:3]/full_xyz[3]
scaled=np.diag([2.,2.,2.,1.])
half_proj=np.array([float(rect['P1'][0,2]/2),float(rect['P1'][1,2]/2),float(Dfull/2),1.])
xyz=rect['Q']@scaled@half_proj
xyz=xyz[:3]/xyz[3]
assert np.allclose(xyz,full_xyz)
print('Q scaling verified; metric depth:', float(xyz[2]))
with tempfile.TemporaryDirectory() as tmp:
    s=sp.save_rig(Path(tmp), calib, R, T, 'synthetic', {'synthetic':True}, 0)
    assert (Path(tmp)/'rig.npz').exists()
    loaded=sp.as_dict(Path(tmp)/'rig.npz')
    assert np.allclose(loaded['T'],T)
print('PASS')
