import todd_tasks.point_tracking as pt

dataset = pt.datasets.TAPVidDAVISDataset()
for t in dataset:
    visual = pt.TAPVidDAVISVisual(t=t)
    colors = visual.colorize()
    visual.trajectory(colors, 2)
    visual.scatter(colors, 5)
    visual.save_video(f'{t["id_"]}.mp4', fps=12)
