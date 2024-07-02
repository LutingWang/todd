import todd.tasks.point_tracking as pt
from todd import Config

dataset = pt.datasets.TAPVidDAVISDataset(
    access_layer=Config(data_root='data/tap_vid'),
)
for t in dataset:
    visual = pt.TAPVidDAVISVisual(t)
    colors = visual.colorize()
    visual.trajectory(colors, 2)
    visual.scatter(colors, 5)
    visual.save_video(f'{t["id_"]}.mp4', fps=12)
