# Graphical abstract

`ga_assets.py` writes thumbnails of one cloudy test patch (input, bicubic, EDiffSR, ESRGAN, RGB / IR composites, label overlay) to `$WORK_DIR/ga_assets`
(needs `cloudy_split` and `cloudy_split_sr`). `make_ga_pptx.py` assembles the editable PowerPoint slide `$WORK_DIR/graphical_abstract.pptx` from them.
The picture of the VEN&mu;S 5 m reference (`ga_assets/s2v_reference.png`) is a crop of the first row of Figure 4 of the paper; put any 5 m reference crop under this name.
The published slide was edited by hand after the generation.
